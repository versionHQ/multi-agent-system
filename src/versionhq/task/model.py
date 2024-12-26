import json
import threading
import uuid
from concurrent.futures import Future
from hashlib import md5
from typing import Any, Dict, List, Set, Optional, Tuple, Callable, Union, Type
from typing_extensions import Annotated

from pydantic import UUID4, BaseModel, Field, PrivateAttr, field_validator, model_validator, create_model
from pydantic_core import PydanticCustomError

from versionhq._utils.process_config import process_config
from versionhq.task import TaskOutputFormat
from versionhq.tool.model import Tool, ToolCalled


class ResponseField(BaseModel):
    """
    Field class to use in the response schema for the JSON response.
    """

    title: str = Field(default=None)
    type: Type = Field(default=str)
    required: bool = Field(default=True)


    def _annotate(self, value: Any) -> Annotated:
        """
        Address `create_model`
        """
        return Annotated[self.type, value] if isinstance(value, self.type) else Annotated[str, str(value)]


    def _convert(self, value: Any) -> Any:
        try:
            if self.type is Any:
                pass
            elif self.type is int:
                return int(value)
            elif self.type is float:
                return float(value)
            elif self.type is list or self.type is dict:
                return json.loads(value)
            else:
                return value
        except:
            return value


    def create_pydantic_model(self, result: Dict, base_model: Union[BaseModel | Any]) -> Any:
        for k, v in result.items():
            if k is not self.title:
                pass
            elif type(v) is not self.type:
                v = self._convert(v)
                setattr(base_model, k, v)
            else:
                setattr(base_model, k, v)
        return base_model


class AgentOutput(BaseModel):
    """
    Keep adding agents' learning and recommendation and store it in `pydantic` field of `TaskOutput` class.
    Since the TaskOutput class has `agent` field, we don't add any info on the agent that handled the task.
    """
    customer_id: str = Field(default=None, max_length=126, description="customer uuid")
    customer_analysis: str = Field(default=None, max_length=256, description="analysis of the customer")
    product_overview: str = Field(default=None, max_length=256, description="analysis of the client's business")
    usp: str = Field()
    cohort_timeframe: int = Field(default=None, max_length=256, description="suitable cohort timeframe in days")
    kpi_metrics: List[str] = Field(default=list, description="Ideal KPIs to be tracked")
    assumptions: List[Dict[str, Any]] = Field(default=list, description="assumptions to test")



class TaskOutput(BaseModel):
    """
    Store the final output of the task in TaskOutput class.
    Depending on the task output format, use `raw`, `pydantic`, `json_dict` accordingly.
    """

    task_id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True, description="store Task ID")
    raw: str = Field(default="", description="Raw output of the task")
    json_dict: Union[Dict[str, Any]] = Field(default=None, description="`raw` converted to dictionary")
    pydantic: Optional[Any] = Field(default=None, description="`raw` converted to the abs. pydantic model")

    def __str__(self) -> str:
        return str(self.pydantic) if self.pydantic else str(self.json_dict) if self.json_dict else self.raw

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


    def to_dict(self) -> Dict[str, Any]:
        """
        Convert pydantic / raw output into dict and return the dict.
        When we only have `raw` output, return `{ output: raw }` to avoid an error
        """

        output_dict = {}
        if self.json_dict:
            output_dict.update(self.json_dict)
        elif self.pydantic:
            output_dict.update(self.pydantic.model_dump())
        else:
            output_dict.upate({ "output": self.raw })
        return output_dict


    def context_prompting(self) -> str:
        """
        When the task is called as context, return its output in concise string to add it to the prompt
        """
        return json.dumps(self.json_dict) if self.json_dict else self.raw[0: 127]



class Task(BaseModel):
    """
    Task to be executed by the agent or the team.
    Each task must have a description and at least one expected output format either Pydantic, Raw, or JSON, with necessary fields in ResponseField.
    Then output will be stored in TaskOutput class.
    """

    __hash__ = object.__hash__

    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True, description="unique identifier for the object, not set by user")
    name: Optional[str] = Field(default=None)
    description: str = Field(description="Description of the actual task")
    _original_description: str = PrivateAttr(default=None)

    # output
    expected_output_json: bool = Field(default=True)
    expected_output_pydantic: bool = Field(default=False)
    output_field_list: List[ResponseField] = Field(
        default=[ResponseField(title="output", type=str, required=False)],
        description="provide output key and data type. this will be cascaded to the agent via task.prompt()"
    )
    output: Optional[TaskOutput] = Field(default=None, description="store the final task output in TaskOutput class")

    # task setup
    context: Optional[List["Task"]] = Field(default=None, description="other tasks whose outputs should be used as context")
    tools_called: Optional[List[ToolCalled]] = Field(default_factory=list, description="tools that the agent can use for this task")
    take_tool_res_as_final: bool = Field(default=False,description="when set True, tools res will be stored in the `TaskOutput`")

    prompt_context: Optional[str] = Field(default=None)
    async_execution: bool = Field(default=False,description="whether the task should be executed asynchronously or not")
    config: Optional[Dict[str, Any]] = Field(default=None, description="configuration for the agent")
    callback: Optional[Any] = Field(default=None, description="callback to be executed after the task is completed.")

    # recording
    processed_by_agents: Set[str] = Field(default_factory=set)
    used_tools: int = 0
    tools_errors: int = 0
    delegations: int = 0


    @property
    def output_prompt(self) -> str:
        """
        Draft prompts on the output format by converting `output_field_list` to dictionary.
        """

        output_prompt, output_formats_to_follow = "", dict()
        for item in self.output_field_list:
            output_formats_to_follow[item.title] = f"<Return your answer in {item.type.__name__}>"

        output_prompt = f"""
Your outputs MUST adhere to the following format and should NOT include any irrelevant elements:
{output_formats_to_follow}
        """
        return output_prompt


    @property
    def expected_output_formats(self) -> List[TaskOutputFormat]:
        """
        Return output formats in list with the ENUM item.
        `TaskOutputFormat.RAW` is set as default.
        """
        outputs = [TaskOutputFormat.RAW,]
        if self.expected_output_json:
            outputs.append(TaskOutputFormat.JSON)
        if self.expected_output_pydantic:
            outputs.append(TaskOutputFormat.PYDANTIC)
        return outputs


    @property
    def key(self) -> str:
        output_format = (
            TaskOutputFormat.JSON
            if self.expected_output_json == True
            else (
                TaskOutputFormat.PYDANTIC
                if self.expected_output_pydantic == True
                else TaskOutputFormat.RAW
            )
        )
        source = [self.description, output_format]
        return md5("|".join(source).encode(), usedforsecurity=False).hexdigest()


    @property
    def summary(self) -> str:
        return f"""
        Task: {self.id} - {self.description}
        "task_description": {self.description}
        "task_expected_output": {self.output_prompt}
        "task_tools": {", ".join([tool_called.tool.name for tool_called in self.tools_called])}
        """


    # validators
    @model_validator(mode="before")
    @classmethod
    def process_model_config(cls, values: Dict[str, Any]):
        return process_config(values_to_update=values, model_class=cls)


    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError("may_not_set_field", "This field is not to be set by the user.", {})


    @model_validator(mode="after")
    def validate_required_fields(self):
        required_fields = ["description",]
        for field in required_fields:
            if getattr(self, field) is None:
                raise ValueError( f"{field} must be provided either directly or through config")
        return self


    @model_validator(mode="after")
    def set_attributes_based_on_config(self) -> "Task":
        """
        Set attributes based on the agent configuration.
        """

        if self.config:
            for key, value in self.config.items():
                setattr(self, key, value)
        return self


    ## comment out as we set raw as the default TaskOutputFormat
    # @model_validator(mode="after")
    # def validate_output_format(self):
    #     if self.expected_output_json == False  and self.expected_output_pydantic == False:
    #         raise PydanticCustomError("Need to choose at least one output format.")
    #     return self


    @model_validator(mode="after")
    def backup_description(self):
        if self._original_description == None:
            self._original_description = self.description
        return self


    def prompt(self, customer: str = None, product_overview: str = None) -> str:
        """
        Format the task prompt and cascade it to the agent.
        When the task has context, add context prompting of all the tasks in the context.
        When we have cusotmer/product info, add them to the prompt.
        """

        task_slices = [self.description, f"{self.output_prompt}", f"Take the following context into consideration: "]

        if self.context:
            context_outputs = "\n".join([task.output.context_prompting() if hasattr(task, "output") else "" for task in self.context])
            task_slices.insert(len(task_slices),  context_outputs)

        if customer:
            task_slices.insert(len(task_slices),  f"Customer overview: {customer}")

        if product_overview:
            task_slices.insert(len(task_slices), f"Product overview: {product_overview}")

        if self.prompt_context:
            task_slices.insert(len(task_slices), self.prompt_context)

        return "\n".join(task_slices)


    def create_json_output(self, raw_result: Any) -> Optional[Dict[str, Any]]:
        """
        Create json (dict) output from the raw result.
        """

        output_json_dict: Optional[Dict[str, Any]] = None

        if isinstance(raw_result, BaseModel):
            output_json_dict = raw_result.model_dump()

        elif isinstance(raw_result, dict):
            output_json_dict = raw_result

        elif isinstance(raw_result, str):
            try:
                output_json_dict = json.loads(raw_result)
            except json.JSONDecodeError:
                try:
                    output_json_dict = eval(raw_result)
                except:
                    try:
                        import ast
                        output_json_dict = ast.literal_eval(raw_result)
                    except:
                        output_json_dict = { "output": raw_result }

        return output_json_dict



    def create_pydantic_output(self, output_json_dict: Dict[str, Any], raw_result: Any = None) -> Optional[Any]:
        """
        Create pydantic output from the `raw` result.
        """

        output_pydantic = None
        if isinstance(raw_result, BaseModel):
            output_pydantic = raw_result

        elif hasattr(output_json_dict, "output"):
            output_pydantic = create_model("PydanticTaskOutput", output=output_json_dict["output"], __base__=BaseModel)

        else:
            output_pydantic = create_model("PydanticTaskOutput", __base__=BaseModel)
            try:
                for item in self.output_field_list:
                    value = output_json_dict[item.title] if hasattr(output_json_dict, item.title) else None
                    if value and type(value) is not item.type:
                        value = item._convert(value)
                    setattr(output_pydantic, item.title, value)
            except:
                setattr(output_pydantic, "output", output_json_dict)

        return output_pydantic


    def _get_output_format(self) -> TaskOutputFormat:
        if self.output_json == True:
            return TaskOutputFormat.JSON
        if self.output_pydantic == True:
            return TaskOutputFormat.PYDANTIC
        return TaskOutputFormat.RAW


    def interpolate_inputs(self, inputs: Dict[str, Any]) -> None:
        """
        Interpolate inputs into the task description and expected output.
        """
        if inputs:
            self.description = self._original_description.format(**inputs)
            # self.expected_output = self._original_expected_output.format(**inputs)


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
        threading.Thread(
            daemon=True,
            target=self._execute_task_async,
            args=(agent, context, future),
        ).start()
        return future


    def _execute_task_async(self, agent, context: Optional[str], future: Future[TaskOutput]) -> None:
        """Execute the task asynchronously with context handling."""
        result = self._execute_core(agent, context)
        future.set_result(result)


    def _execute_core(self, agent, context: Optional[str]) -> TaskOutput:
        """
        Run the core execution logic of the task.
        """

        self.prompt_context = context
        output_raw = agent.execute_task(task=self, context=context)
        output_json_dict = self.create_json_output(raw_result=output_raw)
        output_pydantic = self.create_pydantic_output(output_json_dict=output_json_dict)
        task_output = TaskOutput(
            task_id=self.id,
            raw=output_raw,
            pydantic=output_pydantic,
            json_dict=output_json_dict
        )
        self.output = task_output
        self.processed_by_agents.add(agent.role)

        # self._set_end_execution_time(start_time)

        if self.callback:
            self.callback(self.output)

        # if self._execution_span:
        #     # self._telemetry.task_ended(self._execution_span, self, agent.team)
        #     self._execution_span = None

        # if self.output_file:
        #     content = (
        #         json_output
        #         if json_output
        #         else pydantic_output.model_dump_json() if pydantic_output else result
        #     )
        #     self._save_file(content)
        return task_output


class ConditionalTask(Task):
    """
    A task that can be conditionally executed based on the output of another task.
    Use this with `Team`.
    """

    condition: Callable[[TaskOutput], bool] = Field(
        default=None,
        description="max. number of retries for an agent to execute a task when an error occurs.",
    )

    def __init__(
        self,
        condition: Callable[[Any], bool],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.condition = condition

    def should_execute(self, context: TaskOutput) -> bool:
        """
        Decide whether the conditional task should be executed based on the provided context.
        Return `True` if it should be executed.
        """
        return self.condition(context)

    def get_skipped_task_output(self):
        return TaskOutput(task_id=self.id, raw="", pydantic=None, json_dict=None)
