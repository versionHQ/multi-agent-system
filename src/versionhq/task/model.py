import json
import threading
import uuid
from concurrent.futures import Future
from hashlib import md5
from typing import Any, Dict, List, Set, Optional, Tuple, Callable

from pydantic import (
    UUID4,
    BaseModel,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from versionhq._utils.process_config import process_config
from versionhq.task import TaskOutputFormat
from versionhq.tool.model import Tool, ToolCalled


class ResponseField(BaseModel):
    """
    Field class to use in the response schema for the JSON response.
    """

    title: str = Field(default=None)
    type: str = Field(default=None)
    required: bool = Field(default=True)


class TaskOutput(BaseModel):
    """
    Store the final output of the task in TaskOutput class.
    Depending on the task output format, use `raw`, `pydantic`, `json_dict` accordingly.
    """

    class AgentOutput(BaseModel):
        """
        Keep adding agents' learning and recommendation and store it in `pydantic` field of `TaskOutput` class.
        Since the TaskOutput class has `agent` field, we don't add any info on the agent that handled the task.
        """

        customer_id: str = Field(
            default=None, max_length=126, description="customer uuid"
        )
        customer_analysis: str = Field(
            default=None, max_length=256, description="analysis of the customer"
        )
        business_overview: str = Field(
            default=None,
            max_length=256,
            description="analysis of the client's business",
        )
        cohort_timeframe: int = Field(
            default=None,
            max_length=256,
            description="Suitable cohort timeframe in days",
        )
        kpi_metrics: List[str] = Field(
            default=list, description="Ideal KPIs to be tracked"
        )
        assumptions: List[Dict[str, Any]] = Field(
            default=list, description="assumptions to test"
        )

    task_id: UUID4 = Field(
        default_factory=uuid.uuid4, frozen=True, description="store Task ID"
    )
    raw: str = Field(default="", description="Raw output of the task")
    pydantic: Optional[BaseModel | AgentOutput] = Field(
        default=None, description="Pydantic output of task"
    )
    json_dict: Optional[Dict[str, Any]] = Field(
        default=None, description="JSON dictionary of task"
    )

    def __str__(self) -> str:
        return (
            str(self.pydantic)
            if self.pydantic
            else str(self.json_dict) if self.json_dict else self.raw
        )

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
        """Convert json_output and pydantic_output to a dictionary."""
        output_dict = {}
        if self.json_dict:
            output_dict.update(self.json_dict)
        elif self.pydantic:
            output_dict.update(self.pydantic.model_dump())
        return output_dict


class Task(BaseModel):
    """
    Task to be executed by the agent or the team.
    Each task must have a description and at least one expected output format either Pydantic, Raw, or JSON, with necessary fields in ResponseField.
    Then output will be stored in TaskOutput class.
    """

    __hash__ = object.__hash__

    id: UUID4 = Field(
        default_factory=uuid.uuid4,
        frozen=True,
        description="unique identifier for the object, not set by user",
    )
    name: Optional[str] = Field(default=None)
    description: str = Field(description="Description of the actual task")
    _original_description: str = PrivateAttr(default=None)

    # output
    expected_output_raw: bool = Field(default=False)
    expected_output_json: bool = Field(default=True)
    expected_output_pydantic: bool = Field(default=False)
    output_field_list: Optional[List[ResponseField]] = Field(
        default=[
            ResponseField(title="output", type="str", required=True),
        ]
    )
    output: Optional[TaskOutput] = Field(
        default=None, description="store the final task output in TaskOutput class"
    )

    # task setup
    context: Optional[List["Task"]] = Field(
        default=None, description="other tasks whose outputs should be used as context"
    )
    tools_called: Optional[List[ToolCalled]] = Field(
        default_factory=list, description="tools that the agent can use for this task"
    )
    take_tool_res_as_final: bool = Field(
        default=False,
        description="when set True, tools res will be stored in the `TaskOutput`",
    )

    prompt_context: Optional[str] = None
    async_execution: bool = Field(
        default=False,
        description="whether the task should be executed asynchronously or not",
    )
    config: Optional[Dict[str, Any]] = Field(
        default=None, description="configuration for the agent"
    )
    callback: Optional[Any] = Field(
        default=None, description="callback to be executed after the task is completed."
    )

    # recording
    processed_by_agents: Set[str] = Field(default_factory=set)
    used_tools: int = 0
    tools_errors: int = 0
    delegations: int = 0

    @property
    def output_prompt(self):
        """
        Draft prompts on the output format by converting `output_field_list` to dictionary.
        """

        output_prompt, output_dict = "", dict()
        for item in self.output_field_list:
            output_dict[item.title] = f"your answer in {item.type}"

        output_prompt = f"""
        The output formats include the following format:
        {output_dict}
        """
        return output_prompt

    @property
    def expected_output_formats(self) -> List[TaskOutputFormat]:
        outputs = []
        if self.expected_output_json:
            outputs.append(TaskOutputFormat.JSON)
        if self.expected_output_pydantic:
            outputs.append(TaskOutputFormat.PYDANTIC)
        if self.expected_output_raw:
            outputs.append(TaskOutputFormat.RAW)
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
            raise PydanticCustomError(
                "may_not_set_field", "This field is not to be set by the user.", {}
            )

    @model_validator(mode="after")
    def validate_required_fields(self):
        required_fields = [
            "description",
        ]
        for field in required_fields:
            if getattr(self, field) is None:
                raise ValueError(
                    f"{field} must be provided either directly or through config"
                )
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

    @model_validator(mode="after")
    def validate_output_format(self):
        if (
            self.expected_output_json == False
            and self.expected_output_pydantic == False
            and self.expeceted_output_raw == False
        ):
            raise PydanticCustomError("Need to choose at least one output format.")
        return self

    @model_validator(mode="after")
    def backup_description(self):
        if self._original_description == None:
            self._original_description = self.description
        return self

    def prompt(self, customer=str | None, product_overview=str | None) -> str:
        """
        Return the prompt of the task.
        """

        task_slices = [
            self.description,
            f"Customer overview: {customer}",
            f"Product overview: {product_overview}",
            f"Follow the output formats decribled below. Your response should NOT contain any other element from the following formats.: {self.output_prompt}",
        ]
        return "\n".join(task_slices)

    def _export_output(
        self, result: Any
    ) -> Tuple[Optional[BaseModel], Optional[Dict[str, Any]]]:
        output_pydantic: Optional[BaseModel] = None
        output_json: Optional[Dict[str, Any]] = None
        dict_output = None

        if isinstance(result, str):
            try:
                dict_output = json.loads(result)
            except json.JSONDecodeError:
                try:
                    dict_output = eval(result)
                except:
                    try:
                        import ast

                        dict_output = ast.literal_eval(result)
                    except:
                        dict_output = None

        if self.expected_output_json:
            if isinstance(result, dict):
                output_json = result
            elif isinstance(result, BaseModel):
                output_json = result.model_dump()
            else:
                output_json = dict_output

        if self.expected_output_pydantic:
            if isinstance(result, BaseModel):
                output_pydantic = result
            elif isinstance(result, dict):
                output_json = result
            else:
                output_pydantic = None

        return output_json, output_pydantic

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
        """
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

    def _execute_task_async(
        self, agent, context: Optional[str], future: Future[TaskOutput]
    ) -> None:
        """Execute the task asynchronously with context handling."""
        result = self._execute_core(agent, context)
        future.set_result(result)

    def _execute_core(self, agent, context: Optional[str]) -> TaskOutput:
        """
        Run the core execution logic of the task.
        """

        self.prompt_context = context
        result = agent.execute_task(task=self, context=context)
        output_json, output_pydantic = self._export_output(result)
        task_output = TaskOutput(
            task_id=self.id,
            raw=result,
            pydantic=output_pydantic,
            json_dict=output_json,
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
