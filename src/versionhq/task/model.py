import json
import threading
import datetime
import uuid
import inspect
import enum
from textwrap import dedent
from concurrent.futures import Future
from hashlib import md5
from typing import Any, Dict, List, Set, Optional, Callable, Type
from typing_extensions import Annotated, Self

from pydantic import UUID4, BaseModel, Field, PrivateAttr, field_validator, model_validator, InstanceOf, field_validator
from pydantic_core import PydanticCustomError

import versionhq as vhq
from versionhq.task.evaluation import Evaluation, EvaluationItem
from versionhq.tool.model import Tool, ToolSet
from versionhq._utils import process_config, Logger, is_valid_url


class TaskExecutionType(enum.Enum):
    """
    Enumeration to store task execution types of independent tasks without dependencies.
    """
    SYNC = 1
    ASYNC = 2


class ResponseField(BaseModel):
    """
    A class to store a response format that will generate a JSON schema.
    One layer of nested child is acceptable.
    """

    title: str = Field(default=None, description="title of the field")
    data_type: Type = Field(default=None)
    items: Optional[Type] = Field(default=None, description="store data type of the array items")
    properties: Optional[List[BaseModel]] = Field(default=None, description="store dict items in ResponseField format")
    required: bool = Field(default=True)
    nullable: bool = Field(default=False)
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="additional rules")


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
        Structure valid properties from the ResponseField object. 1 layer of nested child is accepted.
        """
        from versionhq.llm.llm_vars import SchemaType

        schema_type = SchemaType(type=self.data_type).convert()
        props: Dict[str, Any] = {}

        if self.data_type is list:
            if self.items is dict:
                nested_p, nested_r = dict(), list()

                if self.properties:
                    for item in self.properties:
                        nested_p.update(**item._format_props())
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

            elif self.items is list:
                props = {
                    "type": schema_type,
                    "items": { "type": SchemaType(type=self.items).convert(), "items": { "type": SchemaType(type=str).convert() }},
                }

            else:
                props = {
                    "type": schema_type,
                    "items": { "type": SchemaType(type=self.items).convert() },
                }


        elif self.data_type is dict:
            p, r = dict(), list()

            if self.properties:
                for item in self.properties:
                    p.update(**item._format_props())
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


    def _annotate(self, value: Any) -> Annotated:
        """
        Address Pydantic's `create_model`
        """
        return Annotated[self.type, value] if isinstance(value, self.type) else Annotated[str, str(value)]


    def create_pydantic_model(self, result: Dict, base_model: InstanceOf[BaseModel] | Any) -> Any:
        """
        Create a Pydantic model from the given result.
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


class TaskOutput(BaseModel):
    """
    A class to store the final output of the given task in raw (string), json_dict, and pydantic class formats.
    """
    _tokens: int = PrivateAttr(default=0)

    task_id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True, description="store Task ID")
    raw: str = Field(default="", description="Raw output of the task")
    json_dict: Dict[str, Any] = Field(default=None, description="`raw` converted to dictionary")
    pydantic: Optional[Any] = Field(default=None)
    tool_output: Optional[Any] = Field(default=None, description="stores tool result when the task takes tool output as its final output")
    callback_output: Optional[Any] = Field(default=None, description="stores task or agent callback outcome")
    latency: float = Field(default=None, description="job latency in ms")
    evaluation: Optional[InstanceOf[Evaluation]] = Field(default=None, description="stores overall evaluation of the task output. stored in ltm")


    def _to_context_prompt(self) -> str:
        """
        Returns response in string as a prompt context.
        """
        context = ""
        try:
            context = json.dumps(self.json_dict)
        except:
            try:
                if self.pydantic:
                    context = self.pydantic.model_dump()
            except:
                context = self.raw
        return context


    def evaluate(self, task) -> Evaluation:
        """
        Evaluate the output based on the criteria, score each from 0 to 1 scale, and raise suggestions for future improvement.
        """
        from versionhq.task.TEMPLATES.Description import EVALUATE, SHOTS

        self.evaluation = Evaluation() if not self.evaluation else self.evaluation

        eval_criteria = task.eval_criteria if task.eval_criteria else  ["accuracy", "completeness", "conciseness", ]
        fsl_prompt = ""

        if task.fsls:
            fsl_prompt = SHOTS.format(c=task.fsls[0], w=task.fsls[1] if len(task.fsls) > 1 else "")
        else:
            fsl_prompt = self.evaluation._draft_fsl_prompt(task_description=task.description)

        for item in eval_criteria:
            description = EVALUATE.format(task_description=task.description, task_output=self.raw, eval_criteria=str(item))
            description = description + fsl_prompt if fsl_prompt else description

            task_eval = Task(description=description, pydantic_output=EvaluationItem)
            res = task_eval.execute(agent=self.evaluation.eval_by)
            self._tokens += task_eval._tokens

            if res.pydantic:
                item = EvaluationItem(
                    score=res.pydantic.score,
                    weight=res.pydantic.weight,
                    suggestion=res.pydantic.suggestion,
                    criteria=res.pydantic.criteria
                )
                self.evaluation.items.append(item)

            else:
                try:
                    item = EvaluationItem(
                        score=float(res.json_dict["score"]),
                        weight=float(res.json_dict["weight"]),
                        suggestion=res.json_dict["suggestion"],
                        criteria=res.json_dict["criteria"]
                    )
                    self.evaluation.items.append(item)
                except Exception as e:
                    Logger(verbose=True).log(level="error", message=f"Failed to convert the evaluation items: {str(e)}", color="red")
                    pass

        return self.evaluation


    @property
    def aggregate_score(self) -> float | int:
        return self.evaluation.aggregate_score if self.evaluation is not None else 0


    @property
    def json_string(self) -> Optional[str]:
        return json.dumps(self.json_dict)


    def __str__(self) -> str:
        return str(self.pydantic) if self.pydantic else str(self.json_dict) if self.json_dict else self.raw


class Task(BaseModel):
    """
    A class that stores independent task information and handles task executions.
    """

    __hash__ = object.__hash__
    _original_description: str = PrivateAttr(default=None)
    config: Optional[Dict[str, Any]] = Field(default=None, description="values to set on Task class")

    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True, description="unique identifier for the object, not set by user")
    name: Optional[str] = Field(default=None)
    description: str = Field(description="Description of the actual task")

    # response format
    pydantic_output: Optional[Type[BaseModel]] = Field(default=None, description="store Pydantic class as structured response format")
    response_fields: Optional[List[ResponseField]] = Field(default_factory=list, description="store list of ResponseField as structured response format")

    # tool usage
    tools: Optional[List[ToolSet | Tool | Any]] = Field(default_factory=list, description="tools that the agent can use aside from their tools")
    can_use_agent_tools: bool = Field(default=True, description="whether the agent can use their own tools when executing the task")
    tool_res_as_final: bool = Field(default=False, description="when set True, tools res will be stored in the `TaskOutput`")

    image: Optional[str] = Field(default=None, description="absolute file path or url in string")
    file: Optional[str] = Field(default=None, description="absolute file path or url in string")
    audio: Optional[str] = Field(default=None,  description="absolute file path or url in string")

    # executing
    execution_type: TaskExecutionType = Field(default=TaskExecutionType.SYNC)
    allow_delegation: bool = Field(default=False, description="whether to delegate the task to another agent")
    callback: Optional[Callable] = Field(default=None, description="callback to be executed after the task is completed.")
    callback_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="kwargs for the callback when the callback is callable")

    # evaluation
    should_evaluate: bool = Field(default=False, description="True to run the evaluation flow")
    eval_criteria: Optional[List[str]] = Field(default_factory=list, description="stores a list of criteria to evaluate the outcome")
    fsls: Optional[list[str]] = Field(default=None, description="stores ideal/weak responses")

    # recording
    _tokens: int = 0
    _tool_errors: int = 0
    _format_errors: int = 0
    _delegations: int = 0
    processed_agents: Set[str] = Field(default_factory=set, description="store keys of the agents that executed the task")
    output: Optional[TaskOutput] = Field(default=None, description="store the final TaskOutput object")


    @model_validator(mode="before")
    @classmethod
    def process_config(cls, values: Dict[str, Any]) -> None:
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


    def _draft_output_prompt(self, model_provider: str = None) -> str:
        output_prompt = ""

        if self.pydantic_output:
            output_prompt, output_formats_to_follow = "", dict()
            response_format = str(self._structure_response_format(model_provider=model_provider))
            for k, v in self.pydantic_output.model_fields.items():
                output_formats_to_follow[k] = f"<Return your answer in {v.annotation}>"

            output_prompt = f"""Your response MUST be a valid JSON string that strictly follows the response format. Use double quotes for all keys and string values. Do not use single quotes, trailing commas, or any other non-standard JSON syntax.
Response format: {response_format}
Ref. Output image: {output_formats_to_follow}
"""
        elif self.response_fields:
            output_prompt, output_formats_to_follow = "", dict()
            response_format = str(self._structure_response_format(model_provider=model_provider))
            for item in self.response_fields:
                if item:
                    output_formats_to_follow[item.title] = f"<Return your answer in {item.data_type.__name__}>"

            output_prompt = f"""Your response MUST be a valid JSON string that strictly follows the response format. Use double quotes for all keys and string values. Do not use single quotes, trailing commas, or any other non-standard JSON syntax.
Response format: {response_format}
Ref. Output image: {output_formats_to_follow}
"""
        # elif not self.tools or self.can_use_agent_tools == False:
        else:
            output_prompt = "You MUST return your response as a valid JSON serializable string, enclosed in double quotes. Use double quotes for all keys and string values. Do NOT use single quotes, trailing commas, or other non-standard JSON syntax."

        # else:
        #     output_prompt = "You will return a response in a concise manner."

        return dedent(output_prompt)


    def _draft_context_prompt(self, context: Any) -> str:
        """
        Create a context prompt from the given context in any format: a task object, task output object, list, dict.
        """

        context_to_add = None
        if not context:
            # Logger().log(level="error", color="red", message="Missing a context to add to the prompt. We'll return ''.")
            return context_to_add

        match context:
            case str():
                context_to_add = context

            case Task():
                if not context.output:
                    res = context.execute()
                    context_to_add = res._to_context_prompt()

                else:
                    context_to_add = context.output.raw

            case TaskOutput():
                context_to_add = context._to_context_prompt()


            case dict():
                context_to_add = str(context)

            case list():
                res = ", ".join([self._draft_context_prompt(context=item) for item in context])
                context_to_add = res

            case _:
                pass

        return dedent(context_to_add)


    def _user_prompt(self, model_provider: str = None, context: Optional[Any] = None) -> str:
        """
        Format the task prompt and cascade it to the agent.
        """
        output_prompt = self._draft_output_prompt(model_provider=model_provider)
        task_slices = [self.description, output_prompt, ]

        if context:
            context_prompt = self._draft_context_prompt(context=context)
            task_slices.insert(len(task_slices), f"Consider the following context when responding: {context_prompt}")

        return "\n".join(task_slices)


    def _format_content_prompt(self) -> Dict[str, str]:
        """Formats content (file, image, audio) prompts that added to the messages sent to the LLM."""

        from pathlib import Path
        import base64

        content_messages = {}

        if self.image:
            with open(self.image, "rb") as file:
                content = file.read()
                if content:
                    encoded_file = base64.b64encode(content).decode("utf-8")
                    img_url = f"data:image/jpeg;base64,{encoded_file}"
                    content_messages.update({ "type": "image_url", "image_url": { "url": img_url }})

        if self.file:
            if is_valid_url(self.file):
                content_messages.update({ "type": "image_url", "image_url": self.file })

        if self.audio:
            audio_bytes = Path(self.audio).read_bytes()
            encoded_data = base64.b64encode(audio_bytes).decode("utf-8")
            content_messages.update({  "type": "image_url", "image_url": "data:audio/mp3;base64,{}".format(encoded_data)})

        return content_messages


    def _structure_response_format(self, data_type: str = "object", model_provider: str = "gemini") -> Dict[str, Any] | None:
        """Structures `response_fields` or `pydantic_output` to a LLM response format."""

        from versionhq.task.structured_response import StructuredOutput

        response_format: Dict[str, Any] = None

        if model_provider == "openrouter":
            return response_format

        else:
            if self.response_fields:
                properties, required_fields = {}, []
                for i, item in enumerate(self.response_fields):
                    if item:
                        properties.update(item._format_props())
                        required_fields.append(item.title)

                response_schema = {
                    "type": data_type,
                    "properties": properties,
                    "required": required_fields,
                    "additionalProperties": False,
                }
                response_format = {
                    "type": "json_schema",
                    "json_schema": { "name": "outcome", "schema": response_schema }
                }


            elif self.pydantic_output:
                response_format = StructuredOutput(response_format=self.pydantic_output, provider=model_provider)._format()

            return response_format


    def _sanitize_raw_output(self, raw: str) -> Dict[str, str]:
        """Sanitizes raw output and prepare for json.loads"""

        import re
        import ast

        output, j = None, None
        r = str(raw).strip()
        r = r.replace("true", "True").replace("false", "False").replace("```json", '"').replace("```", '"').replace('\n', '').replace('\\', '')
        r = re.sub("^'", '"', r)
        r = re.sub(r"'\b", '"', r)
        r = r.strip()
        r = r.replace("  ", "")
        try:
            output = json.loads(r)
        except:
            try: j = json.dumps(eval(r))
            except:
                try: j = json.dumps(str(r))
                except: j = r
            output = json.loads(j)

        if isinstance(output, dict):
            return output
        else:
            try:
                output = ast.literal_eval(j)
            except:
                output = ast.literal_eval(r)

            return output if isinstance(output, dict) else { "output": str(r) }


    def _create_json_output(self, raw: str) -> Dict[str, Any]:
        """Creates JSON output from the raw output."""

        output = None

        if raw is None or raw == "":
            Logger().log(level="warning", message="The model returned an empty response. Returning an empty dict.", color="yellow")
            output = { "output": "" }
            return output

        try:
            output = json.loads(raw)
            if isinstance(output, dict):
                return output
            else:
               output = self._sanitize_raw_output(raw=raw)
               return output
        except:
            output = self._sanitize_raw_output(raw=raw)
            return output


    def _create_pydantic_output(self, raw: str = None, json_dict: Dict[str, Any] = None) -> InstanceOf[BaseModel]:
        """
        Create pydantic output from raw or json_dict output.
        """

        output_pydantic = self.pydantic_output

        try:
            json_dict = json_dict if json_dict else self._create_json_output(raw=raw)

            for k, v in json_dict.items():
                setattr(output_pydantic, k, v)
        except:
            pass

        return output_pydantic


    def interpolate_inputs(self, inputs: Dict[str, Any]) -> None:
        """
        Interpolate inputs into the task description.
        """
        self._original_description = self.description

        if inputs:
            self.description = self._original_description.format(**inputs)


    def _create_short_and_long_term_memories(self, agent: Any, task_output: TaskOutput) -> None:
        """
        After the task execution, create and save short-term/long-term memories in the storage.
        """
        from versionhq.agent.model import Agent
        from versionhq.memory.model import ShortTermMemory, MemoryMetadata, LongTermMemory

        agent = agent if isinstance(agent, Agent) else Agent(role=str(agent), goal=str(agent), with_memory=True)

        if agent.with_memory == False:
            return None

        try:
            evaluation = task_output.evaluation if task_output.evaluation else None
            memory_metadata = evaluation._create_memory_metadata() if evaluation else MemoryMetadata()

            agent.short_term_memory = agent.short_term_memory if agent.short_term_memory else ShortTermMemory(agent=agent, embedder_config=agent.embedder_config)
            agent.short_term_memory.save(
                task_description=str(self.description),
                task_output=str(task_output.raw),
                agent=str(agent.role),
                metadata=memory_metadata
            )

            agent.long_term_memory = agent.long_term_memory if agent.long_term_memory else LongTermMemory()
            agent.long_term_memory.save(
                task_description=str(self.description),
                task_output=str(task_output.raw),
                agent=str(agent.role),
                metadata=memory_metadata
            )

        except AttributeError as e:
            Logger().log(level="error", message=f"Missing attributes for long term memory: {str(e)}", color="red")
            pass

        except Exception as e:
            Logger().log(level="error", message=f"Failed to add to the memory: {str(e)}", color="red")
            pass


    def _build_agent_from_task(self, task_description: str = None) -> InstanceOf["vhq.Agent"]:
        task_description = task_description if task_description else self.description
        if not task_description:
            Logger().log(level="error", message="Task is missing the description.", color="red")
            pass

        agent = vhq.Agent(goal=task_description, role=task_description, maxit=1) #! REFINEME
        return agent


    def _select_agent_to_delegate(self, agent: Any = None) -> Any | None: # return agent object or None
        """
        Creates or selects an agent to delegate the given task and returns Agent object else None.
        """

        from versionhq.agent.model import Agent

        if not self.allow_delegation:
            return None

        agent_to_delegate: InstanceOf[Agent] = None

        if not agent:
            agent_to_delegate = self._build_agent_from_task()

        elif agent and not agent.networks:
            agent_to_delegate = Agent(role="vhq-Delegated-Agent", goal=agent.goal, llm=agent.llm)

        else:
            _managers = []
            _members = []
            for network in agent.networks:
                _managers.extend(member.agent for member in network.members if member.is_manager)
                _members.extend(member.agent for member in network.members if not member.is_manager)

            agent_to_delegate = _managers[0] if _managers else _members[0] if _members else Agent(role="vhq-Delegated-Agent", goal=agent.goal, llm=agent.llm)

        return agent_to_delegate


    def _store_logs(self, inputs: Optional[Dict[str, Any]] = {}) -> None:
        from versionhq.storage.task_output_storage import TaskOutputStorageHandler

        TaskOutputStorageHandler().update(task=self, inputs=inputs)


    # task execution
    def execute(
            self, type: TaskExecutionType = None, agent: Optional["vhq.Agent"] = None, context: Optional[Any] = None
        ) -> TaskOutput | Future[TaskOutput]:
        """
        A main method to handle task execution. Build an agent when the agent is not given.
        """
        type = type if type else  self.execution_type if self.execution_type else TaskExecutionType.SYNC

        if not agent:
            agent = self._build_agent_from_task(task_description=self.description)

        match type:
            case TaskExecutionType.SYNC:
                return self._execute_sync(agent=agent, context=context)

            case TaskExecutionType.ASYNC:
                return self._execute_async(agent=agent, context=context)


    def _execute_sync(self, agent, context: Optional[Any] = None) -> TaskOutput:
        """Executes the task synchronously."""
        return self._execute_core(agent, context)


    def _execute_async(self, agent, context: Optional[Any] = None) -> Future[TaskOutput]:
        """Executes the task asynchronously."""
        future: Future[TaskOutput] = Future()

        def _handle_task_async(self, agent, context: Optional[str], future: Future[TaskOutput]) -> None:
            result = self._execute_core(agent, context)
            future.set_result(result)

        threading.Thread(daemon=True, target=_handle_task_async, args=(agent, context, future)).start()
        return future


    def _execute_core(self, agent, context: Optional[Any]) -> TaskOutput:
        """
        A core method to execute a task.
        """
        task_output: InstanceOf[TaskOutput] = None
        raw_output: str = None
        tool_output: str | list = None
        task_tools: List[List[InstanceOf[Tool]| InstanceOf[ToolSet] | Type[Tool]]] = []
        started_at, ended_at = datetime.datetime.now(), datetime.datetime.now()

        if self.tools:
            for item in self.tools:
                if isinstance(item, ToolSet) or isinstance(item, Tool) or type(item) == Tool:
                    task_tools.append(item)

        if self.allow_delegation == True:
            agent_to_delegate = self._select_agent_to_delegate(agent=agent)
            agent = agent_to_delegate
            self._delegations += 1

        if self.tool_res_as_final == True:
            started_at = datetime.datetime.now()
            tool_output = agent.execute_task(task=self, context=context, task_tools=task_tools)
            raw_output = str(tool_output) if tool_output else ""
            ended_at = datetime.datetime.now()
            task_output = TaskOutput(task_id=self.id, tool_output=tool_output, raw=raw_output)

        else:
            started_at = datetime.datetime.now()
            raw_output = agent.execute_task(task=self, context=context, task_tools=task_tools)
            ended_at = datetime.datetime.now()

            json_dict_output = self._create_json_output(raw=raw_output)
            if "outcome" in json_dict_output:
                json_dict_output = self._create_json_output(raw=str(json_dict_output["outcome"]))

            pydantic_output = self._create_pydantic_output(raw=raw_output, json_dict=json_dict_output) if self.pydantic_output else None

            task_output = TaskOutput(
                task_id=self.id,
                raw=raw_output if raw_output is not None else "",
                pydantic=pydantic_output,
                json_dict=json_dict_output,
            )

        task_output.latency = round((ended_at - started_at).total_seconds() * 1000, 3)
        task_output._tokens = self._tokens
        self.output = task_output
        self.processed_agents.add(agent.key)

        # if self.output_file: ## disabled for now
        #     content = (
        #         json_output
        #         if json_output
        #         else pydantic_output.model_dump_json() if pydantic_output else result
        #     )
        #     self._save_file(content)

        if raw_output:
            if self.should_evaluate:
                task_output.evaluate(task=self)
                self.output = task_output

            self._create_short_and_long_term_memories(agent=agent, task_output=task_output)

            if self.callback and isinstance(self.callback, Callable):
                kwargs = { **self.callback_kwargs, **task_output.json_dict }
                sig = inspect.signature(self.callback)
                valid_keys = [param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
                valid_kwargs = { k: kwargs[k] if  k in kwargs else None for k in valid_keys }
                callback_res = self.callback(**valid_kwargs)
                task_output.callback_output = callback_res
                self.output = task_output
            self._store_logs()

        return task_output


    @property
    def key(self) -> str:
        output_format = "json" if self.response_fields else "pydantic" if self.pydantic_output is not None else "raw"
        source = [self.description, output_format]
        return md5("|".join(source).encode(), usedforsecurity=False).hexdigest()


    @property
    def summary(self) -> str:
        return f"""
Task ID: {str(self.id)}
"Description": {self.description}
"Tools": {", ".join([tool.name for tool in self.tools])}
        """
