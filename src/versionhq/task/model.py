import json
import threading
import datetime
import uuid
from concurrent.futures import Future
from hashlib import md5
from typing import Any, Dict, List, Set, Optional, Tuple, Callable, Type
from typing_extensions import Annotated, Self

from pydantic import UUID4, BaseModel, Field, PrivateAttr, field_validator, model_validator, create_model, InstanceOf
from pydantic_core import PydanticCustomError

from versionhq._utils.process_config import process_config
from versionhq.task import TaskOutputFormat
from versionhq.task.log_handler import TaskOutputStorageHandler
from versionhq.tool.model import Tool, ToolSet
from versionhq._utils.logger import Logger


class ResponseField(BaseModel):
    """
    A class to store the response format and schema that will cascade to the LLM.
    The `config` field can store additional params:
    https://community.openai.com/t/official-documentation-for-supported-schemas-for-response-format-parameter-in-calls-to-client-beta-chats-completions-parse/932422/3
    """

    title: str = Field(default=None, description="title of the field")
    data_type: Type = Field(default=None)
    items: Optional[Type] = Field(default=None, description="store data type of the array items")
    properties: Optional[List[BaseModel]] = Field(default=None, description="store dict items in ResponseField format")
    required: bool = Field(default=True)
    nullable: bool = Field(default=False)
    config: Optional[Dict[str, Any]] = Field(default=None, description="additional rules")


    @model_validator(mode="after")
    def validate_instance(self) -> Self:
        """
        Validate the model instance based on the given `data_type`. (An array must have `items`, dict must have properties.)
        """

        if self.data_type is list and self.items is None:
            self.items = str

        if self.data_type is dict or (self.data_type is list and self.items is dict):
            if self.properties is None:
                raise PydanticCustomError("missing_properties", "The dict type has to set the properties.", {})

            else:
                for item in self.properties:
                    if not isinstance(item, ResponseField):
                        raise PydanticCustomError("invalid_properties", "Properties field must input in ResponseField format.", {})

        return self


    def _format_props(self) -> Dict[str, Any]:
        """
        Structure valid properties. We accept 2 nested objects.
        """
        from versionhq.llm.llm_vars import SchemaType

        schema_type = SchemaType(type=self.data_type).convert()
        props: Dict[str, Any] = {}

        if self.data_type is list and self.items is not dict:
            props = {
                "type": schema_type,
                "items": { "type": SchemaType(type=self.items).convert() },
            }

        elif self.data_type is list and self.items is dict:
            nested_p, nested_r = dict(), list()

            if self.properties:
                for item in self.properties:
                    nested_p.update(**item._format_props())

                    if item.required:
                        nested_r.append(item.title)

            props = {
                "type": schema_type,
                "items": {
                    "type": SchemaType(type=self.items).convert(),
                    "properties": nested_p,
                    "required": nested_r,
                    "additionalProperties": False
                }
            }

        elif self.data_type is dict:
            p, r = dict(), list()

            if self.properties:
                for item in self.properties:
                    p.update(**item._format_props())

                    if item.required:
                        r.append(item.title)

            props = {
                "type": schema_type,
                "properties": p,
                "required": r,
                "additionalProperties": False
            }

        else:
            props = {
                "type": schema_type,
                "nullable": self.nullable,
            }

        return { self.title: { **props, **self.config }} if self.config else { self.title: props }


    def _convert(self, value: Any) -> Any:
        """
        Convert the given value to the ideal data type.
        """
        try:
            if self.type is Any:
                pass
            elif self.type is int:
                return int(value)
            elif self.type is float:
                return float(value)
            elif self.type is list or self.type is dict:
                return json.loads(eval(str(value)))
            elif self.type is str:
                return str(value)
            else:
                return value
        except:
            return value


    def create_pydantic_model(self, result: Dict, base_model: InstanceOf[BaseModel] | Any) -> Any:
        """
        Create a Pydantic model from the given result
        """
        for k, v in result.items():
            if k is not self.title:
                pass
            elif type(v) is not self.type:
                v = self._convert(v)
                setattr(base_model, k, v)
            else:
                setattr(base_model, k, v)
        return base_model


    def _annotate(self, value: Any) -> Annotated:
        """
        Address Pydantic's `create_model`
        """
        return Annotated[self.type, value] if isinstance(value, self.type) else Annotated[str, str(value)]



class TaskOutput(BaseModel):
    """
    Store the final output of the task in TaskOutput class.
    Depending on the task output format, use `raw`, `pydantic`, `json_dict` accordingly.
    """

    task_id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True, description="store Task ID")
    raw: str = Field(default="", description="Raw output of the task")
    json_dict: Dict[str, Any] = Field(default=None, description="`raw` converted to dictionary")
    pydantic: Optional[Any] = Field(default=None, description="`raw` converted to the abs. pydantic model")
    tool_output: Optional[Any] = Field(default=None, description="store tool result when the task takes tool output as its final output")

    def __str__(self) -> str:
        return str(self.pydantic) if self.pydantic else str(self.json_dict) if self.json_dict else self.raw


    def to_dict(self) -> Dict[str, Any] | None:
        """
        Convert pydantic / raw output into dict and return the dict.
        """
        return self.json_dict if self.json_dict is not None else self.pydantic.model_dump() if self.pydantic else None


    def context_prompting(self) -> str:
        """
        When the task is called as context, return its output in concise string to add it to the prompt
        """
        return json.dumps(self.json_dict) if self.json_dict else self.raw[0: 127]


    @property
    def json(self) -> Optional[str]:
        if self.output_format != TaskOutputFormat.JSON:
            raise ValueError(
                """
                Invalid output format requested.
                If you would like to access the JSON output,
                pleae make sure to set the output_json property for the task
                """
            )
        return json.dumps(self.json_dict)



class Task(BaseModel):
    """
    Task to be executed by agents or teams.
    Each task must have a description.
    Default response is JSON string that strictly follows `response_fields` - and will be stored in TaskOuput.raw / json_dict.
    When `pydantic_output` is provided, we prioritize them and store raw (json string), json_dict, pydantic in the TaskOutput class.
    """

    __hash__ = object.__hash__
    _logger: Logger = PrivateAttr(default_factory=lambda: Logger(verbose=True))
    _original_description: str = PrivateAttr(default=None)
    _task_output_handler = TaskOutputStorageHandler()
    config: Optional[Dict[str, Any]] = Field(default=None, description="values to set on Task class")

    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True, description="unique identifier for the object, not set by user")
    name: Optional[str] = Field(default=None)
    description: str = Field(description="Description of the actual task")

    # output
    pydantic_custom_output: Optional[Any] = Field(
        default=None,
        description="store a custom Pydantic class that will be passed to the model as a response format."
    )
    response_fields: List[ResponseField] = Field(
        default_factory=list,
        description="store the list of ResponseFields to create the response format"
    )
    output: Optional[TaskOutput] = Field(default=None, description="store the final task output in TaskOutput class")

    # task setup
    context: Optional[List["Task"]] = Field(default=None, description="other tasks whose outputs should be used as context")
    prompt_context: Optional[str] = Field(default=None)

    # tool usage
    tools: Optional[List[ToolSet | Tool | Any]] = Field(default_factory=list, description="tools that the agent can use aside from their tools")
    can_use_agent_tools: bool = Field(default=False, description="whether the agent can use their own tools when executing the task")
    tool_res_as_final: bool = Field(default=False, description="when set True, tools res will be stored in the `TaskOutput`")

    # execution rules
    allow_delegation: bool = Field(default=False, description="ask other agents for help and run the task instead")
    async_execution: bool = Field(default=False,description="whether the task should be executed asynchronously or not")
    callback: Optional[Any] = Field(default=None, description="callback to be executed after the task is completed.")
    callback_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="kwargs for the callback when the callback is callable")

    # recording
    processed_by_agents: Set[str] = Field(default_factory=set, description="store responsible agents' roles")
    tools_errors: int = 0
    delegations: int = 0
    execution_span_in_sec: int = 0


    @model_validator(mode="before")
    @classmethod
    def process_model_config(cls, values: Dict[str, Any]) -> None:
        return process_config(values_to_update=values, model_class=cls)


    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError("may_not_set_field", "This field is not to be set by the user.", {})


    @model_validator(mode="after")
    def validate_required_fields(self) -> Self:
        required_fields = ["description",]
        for field in required_fields:
            if getattr(self, field) is None:
                raise ValueError( f"{field} must be provided either directly or through config")
        return self


    @model_validator(mode="after")
    def set_attributes_based_on_config(self) -> Self:
        """
        Set attributes based on the task configuration.
        """

        if self.config:
            for key, value in self.config.items():
                setattr(self, key, value)
        return self


    @model_validator(mode="after")
    def set_up_tools(self) -> Self:
        if not self.tools:
            pass
        else:
            tool_list = []
            for item in self.tools:
                if isinstance(item, Tool) or isinstance(item, ToolSet):
                    tool_list.append(item)
                elif (isinstance(item, dict) and "function" not in item) or isinstance(item, str):
                    pass
                else:
                    tool_list.append(item) # address custom tool
            self.tools = tool_list
        return self


    @model_validator(mode="after")
    def backup_description(self):
        if self._original_description == None:
            self._original_description = self.description
        return self


    def _draft_output_prompt(self, model_provider: str) -> str:
        """
        Draft prompts on the output format by converting `
        """

        output_prompt = ""

        if self.pydantic_custom_output:
            output_prompt = f"""
Your response MUST STRICTLY follow the given repsonse format:
JSON schema: {str({k: v for k, v in self.pydantic_custom_output.__fields__.items()})}
"""

        elif self.response_fields:
            output_prompt, output_formats_to_follow = "", dict()
            response_format = str(self._structure_response_format(model_provider=model_provider))
            for item in self.response_fields:
                if item:
                    output_formats_to_follow[item.title] = f"<Return your answer in {item.data_type.__name__}>"

            output_prompt = f"""
Your response MUST be a valid JSON string that strictly follows the response format. Use double quotes for all keys and string values. Do not use single quotes, trailing commas, or any other non-standard JSON syntax.
Response format: {response_format}
Ref. Output image: {output_formats_to_follow}
"""

        else:
            output_prompt = "Return your response as a valid JSON string, enclosed in double quotes. Do not use single quotes, trailing commas, or other non-standard JSON syntax."

        return output_prompt


    def prompt(self, model_provider: str = None, customer: str = None, product_overview: str = None) -> str:
        """
        Format the task prompt and cascade it to the agent.
        When the task has context, add context prompting of all the tasks in the context.
        When we have cusotmer/product info, add them to the prompt.
        """

        output_prompt = self._draft_output_prompt(model_provider=model_provider)
        task_slices = [self.description, output_prompt,]

        if self.context:
            context_outputs = "\n".join([task.output.context_prompting() if hasattr(task, "output") else "" for task in self.context])
            task_slices.insert(len(task_slices), f"Consider the following context when responding: {context_outputs}")

        if self.prompt_context:
            task_slices.insert(len(task_slices), f"Consider the following context when responding: {self.prompt_context}")

        if customer:
            task_slices.insert(len(task_slices), f"Customer to address: {customer}")

        if product_overview:
            task_slices.insert(len(task_slices), f"Product to promote: {product_overview}")

        return "\n".join(task_slices)


    def _get_output_format(self) -> TaskOutputFormat:
        if self.output_json == True:
            return TaskOutputFormat.JSON
        if self.output_pydantic == True:
            return TaskOutputFormat.PYDANTIC
        return TaskOutputFormat.RAW


    def _structure_response_format(self, data_type: str = "object", model_provider: str = "gemini") -> Dict[str, Any] | None:
        """
        Create and return a valid response format using
            - mannual response schema from `self.response_fields`, or
            - SDK objects from `pydantic_custom_output`.
        OpenAI:
        https://platform.openai.com/docs/guides/structured-outputs?context=ex1#function-calling-vs-response-format
        https://platform.openai.com/docs/guides/structured-outputs?context=with_parse#some-type-specific-keywords-are-not-yet-supported
        Gemini:
        """

        response_schema = None

        if self.response_fields:
            properties, required_fields = {}, []
            for i, item in enumerate(self.response_fields):
                if item:
                    if item.data_type is dict:
                        properties.update(item._format_props())
                    else:
                        properties.update(item._format_props())

                    required_fields.append(item.title)

            response_schema = {
                "type": "object",
                "properties": properties,
                "required": required_fields,
                "additionalProperties": False, # for openai
            }


        elif self.pydantic_custom_output:
            response_schema = {
                **self.pydantic_custom_output.model_json_schema(),
                "additionalProperties": False,
                "required": [k for k, v in self.pydantic_custom_output.__fields__.items()],
                "strict": True,
            }


        if response_schema:
            if model_provider == "gemini":
                return {
                    "type": data_type,
                    "response_schema": response_schema,
                    "enforce_validation": True
                }

            if model_provider == "openai":
                if self.pydantic_custom_output:
                    return self.pydantic_custom_output
                else:
                    return {
                        "type": "json_schema",
                        "json_schema": { "name": "outcome", "strict": True, "schema": response_schema },
                    }
        else:
            return None


    def _create_json_output(self, raw: str) -> Dict[str, Any]:
        """
        Create json (dict) output from the raw output and `response_fields` information.
        """

        if raw is None or raw == "":
            self._logger.log(level="error", message="The model returned an empty response. Returning an empty dict.", color="yellow")
            output = { "output": "n.a." }
            return output

        try:
            r = str(raw).replace("true", "True").replace("false", "False")
            j = json.dumps(eval(r))
            output = json.loads(j)
            if isinstance(output, dict):
                return output

            else:
                r = str(raw).replace("{'", '{"').replace("{ '", '{"').replace("': '", '": "').replace("'}", '"}').replace("' }", '"}').replace("', '", '", "').replace("['", '["').replace("[ '", '[ "').replace("']", '"]').replace("' ]", '" ]').replace("{\n'", '{"').replace("{\'", '{"').replace("true", "True").replace("false", "False")
                j = json.dumps(eval(r))
                output = json.loads(j)

                if isinstance(output, dict):
                    return output

                else:
                    import ast
                    output = ast.literal_eval(r)
                    return output if isinstance(output, dict) else { "output": str(r) }

        except:
            output = { "output": str(raw) }
            return output



    def _create_pydantic_output(self, raw: str = None, json_dict: Dict[str, Any] = None) -> InstanceOf[BaseModel]:
        """
        Create pydantic output from the `raw` result.
        """

        output_pydantic = None
        json_dict = json_dict

        try:
            if not json_dict:
                json_dict = self._create_json_output(raw=raw)

            output_pydantic = self.pydantic_custom_output(**json_dict)

        except:
            pass

        return output_pydantic


    def interpolate_inputs(self, inputs: Dict[str, Any]) -> None:
        """
        Interpolate inputs into the task description and expected output.
        """
        if inputs:
            self.description = self._original_description.format(**inputs)


    # task execution
    def execute_sync(self, agent, context: Optional[str] = None) -> TaskOutput:
        """
        Execute the task synchronously.
        When the task has context, make sure we have executed all the tasks in the context first.
        """

        if self.context:
            for task in self.context:
                if task.output is None:
                    task._execute_core(agent, context)

        return self._execute_core(agent, context)


    def execute_async(self, agent, context: Optional[str] = None) -> Future[TaskOutput]:
        """
        Execute the task asynchronously.
        """

        future: Future[TaskOutput] = Future()
        threading.Thread(daemon=True, target=self._execute_task_async, args=(agent, context, future)).start()
        return future


    def _execute_task_async(self, agent, context: Optional[str], future: Future[TaskOutput]) -> None:
        """
        Execute the task asynchronously with context handling.
        """

        result = self._execute_core(agent, context)
        future.set_result(result)


    def _execute_core(self, agent, context: Optional[str]) -> TaskOutput:
        """
        Execute the given task with the given agent.
        Handle 1. agent delegation, 2. tools, 3. context to consider, and 4. callbacks
        """

        from versionhq.agent.model import Agent
        from versionhq.team.model import Team

        self.prompt_context = context
        task_output: InstanceOf[TaskOutput] = None
        tool_output: str | list = None
        task_tools: List[InstanceOf[Tool] | InstanceOf[ToolSet] | Type[Tool]] = []
        started_at = datetime.datetime.now()

        if self.tools:
            for item in self.tools:
                if isinstance(item, ToolSet) or isinstance(item, Tool) or type(item) == Tool:
                    task_tools.append(item)

        if self.allow_delegation:
            agent_to_delegate = None

            if hasattr(agent, "team") and isinstance(agent.team, Team):
                if agent.team.managers:
                    idling_manager_agents = [manager.agent for manager in agent.team.managers if manager.is_idling]
                    agent_to_delegate = idling_manager_agents[0] if idling_manager_agents else agent.team.managers[0]
                else:
                    peers = [member.agent for member in agent.team.members if member.is_manager == False and member.agent.id is not agent.id]
                    if len(peers) > 0:
                        agent_to_delegate = peers[0]
            else:
                agent_to_delegate = Agent(role="delegated_agent", goal=agent.goal, llm=agent.llm)

            agent = agent_to_delegate
            self.delegations += 1


        if self.tool_res_as_final == True:
            tool_output = agent.execute_task(task=self, context=context, task_tools=task_tools)
            task_output = TaskOutput(task_id=self.id, tool_output=tool_output)

        else:
            raw_output = agent.execute_task(task=self, context=context, task_tools=task_tools)
            json_dict_output = self._create_json_output(raw=raw_output)
            if "outcome" in json_dict_output:
                json_dict_output = self._create_json_output(raw=str(json_dict_output["outcome"]))

            pydantic_output = self._create_pydantic_output(raw=raw_output, json_dict=json_dict_output) if self.pydantic_custom_output else None

            task_output = TaskOutput(
                task_id=self.id,
                raw=raw_output,
                pydantic=pydantic_output,
                json_dict=json_dict_output
            )

        self.output = task_output
        self.processed_by_agents.add(agent.role)

        if self.callback:
            self.callback({ **self.callback_kwargs, **self.output.__dict__ })

        # if self.output_file:
        #     content = (
        #         json_output
        #         if json_output
        #         else pydantic_output.model_dump_json() if pydantic_output else result
        #     )
        #     self._save_file(content)
        ended_at = datetime.datetime.now()
        self.execution_span_in_sec = (ended_at - started_at).total_seconds()

        return task_output


    def _store_execution_log(self, task_index: int, was_replayed: bool = False, inputs: Optional[Dict[str, Any]] = {}) -> None:
        """
        Store the task execution log.
        """

        self._task_output_handler.update(task=self, task_index=task_index, was_replayed=was_replayed, inputs=inputs)


    @property
    def key(self) -> str:
        output_format = TaskOutputFormat.JSON if self.response_fields else TaskOutputFormat.PYDANTIC if self.pydantic_custom_output is not None else TaskOutputFormat.RAW
        source = [self.description, output_format]
        return md5("|".join(source).encode(), usedforsecurity=False).hexdigest()


    @property
    def summary(self) -> str:
        return f"""
Task ID: {str(self.id)}
"Description": {self.description}
"Tools": {", ".join([tool.name for tool in self.tools])}
        """



class ConditionalTask(Task):
    """
    A task that can be conditionally executed based on the output of another task.
    When the `condition` return True, execute the task, else skipped with `skipped task output`.
    """

    condition: Callable[[TaskOutput], bool] = Field(
        default=None,
        description="max. number of retries for an agent to execute a task when an error occurs",
    )


    def __init__(self, condition: Callable[[Any], bool], **kwargs):
        super().__init__(**kwargs)
        self.condition = condition
        self._logger = Logger(verbose=True)


    def should_execute(self, context: TaskOutput) -> bool:
        """
        Decide whether the conditional task should be executed based on the provided context.
        Return `True` if it should be executed.
        """
        return self.condition(context)


    def get_skipped_task_output(self):
        return TaskOutput(task_id=self.id, raw="", pydantic=None, json_dict={})


    def _handle_conditional_task(self, task_outputs: List[TaskOutput], task_index: int, was_replayed: bool) -> Optional[TaskOutput]:
        """
        When the conditional task should be skipped, return `skipped_task_output` as task_output else return None
        """

        previous_output = task_outputs[task_index - 1] if task_outputs and len(task_outputs) > 1  else None

        if previous_output and not self.should_execute(previous_output):
            self._logger.log(level="debug", message=f"Skipping conditional task: {self.description}", color="yellow")
            skipped_task_output = self.get_skipped_task_output()
            self.output = skipped_task_output

            if not was_replayed:
                self._store_execution_log(self, task_index=task_index, was_replayed=was_replayed, inputs={})
            return skipped_task_output

        return None
