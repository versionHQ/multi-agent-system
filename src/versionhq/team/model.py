import uuid
import warnings
import json
from abc import ABC
from enum import Enum
from dotenv import load_dotenv
from concurrent.futures import Future
from hashlib import md5
from typing import Any, Dict, List, TYPE_CHECKING, Callable, Optional, Tuple, Union
from pydantic import (
    UUID4,
    InstanceOf,
    Json,
    BaseModel,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from versionhq.agent.model import Agent
from versionhq.task.model import Task, TaskOutput, ConditionalTask, TaskOutputFormat
from versionhq.task.formatter import create_raw_outputs
from versionhq.team.team_planner import TeamPlanner
from versionhq._utils.logger import Logger
from versionhq._utils.usage_metrics import UsageMetrics


from pydantic._internal._generate_schema import GenerateSchema
from pydantic_core import core_schema

initial_match_type = GenerateSchema.match_type


def match_type(self, obj):
    if getattr(obj, "__name__", None) == "datetime":
        return core_schema.datetime_schema()
    return initial_match_type(self, obj)


GenerateSchema.match_type = match_type

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
load_dotenv(override=True)

# agentops = None
# if os.environ.get("AGENTOPS_API_KEY"):
#     try:
#         import agentops  # type: ignore
#     except ImportError:
#         pass


class TaskHandlingProcess(str, Enum):
    """
    Class representing the different processes that can be used to tackle multiple tasks.
    """
    sequential = "sequential"
    hierarchical = "hierarchical"
    consensual = "consensual"


class TeamOutput(BaseModel):
    """
    Store outputs of the tasks handled by the team.
    `json_dict` and `raw` store overall output of tasks that handled by the team,
    while `task_output_list` stores each TaskOutput instance to the tasks handled by the team members.
    Note that `raw` and `json_dict` will be prioritized as TeamOutput to refer over `task_output_list`.
    """

    team_id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True, description="store the team ID that generate the TeamOutput")
    raw: str = Field(default="", description="raw output of the team lead task handled by the team leader")
    pydantic: Optional[Any] = Field(default=None, description="`raw` converted to the abs. pydantic model")
    json_dict: Union[Dict[str, Any]] = Field(default=None, description="`raw` converted to dictionary")
    task_output_list: list[TaskOutput] = Field(default=list, description="store output of all the tasks that the team has executed")
    token_usage: UsageMetrics = Field(default=dict, description="processed token summary")

    def __str__(self):
        return (str(self.pydantic) if self.pydantic else str(self.json_dict) if self.json_dict else self.raw)


    def __getitem__(self, key):
        if self.pydantic and hasattr(self.pydantic, key):
            return getattr(self.pydantic, key)
        elif self.json_dict and key in self.json_dict:
            return self.json_dict[key]
        else:
            raise KeyError(f"Key '{key}' not found in the team output.")


    @property
    def json(self) -> Optional[str]:
        if self.tasks_output[-1].output_format != TaskOutputFormat.JSON:
            raise ValueError(
                "No JSON output found in the final task. Please make sure to set the output_json property in the final task in your team."
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


    def return_all_task_outputs(self) -> List[Dict[str, Any]]:
        res = [output.json_dict for output in self.task_output_list]
        return res


class TeamMember(ABC, BaseModel):
    agent: Agent | None = Field(default=None, description="store the agent to be a member")
    is_manager: bool = Field(default=False)
    task: Task | None = Field(default=None)


class Team(BaseModel):
    """
    A collaborative team of agents that handles complex, multiple tasks.
    We define strategies for task executions and overall workflow.
    """

    __hash__ = object.__hash__
    _execution_span: Any = PrivateAttr()
    _logger: Logger = PrivateAttr()
    # _inputs: Optional[Dict[str, Any]] = PrivateAttr(default=None)

    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    name: Optional[str] = Field(default=None)
    members: List[TeamMember] = Field(default_factory=list, description="store agents' uuids and bool if it is manager")

    # work as a team
    team_tasks: Optional[List[Task]] = Field(default_factory=list, description="optional tasks for the team")
    planning_llm: Optional[Any] = Field(default=None, description="llm to handle the planning of the team tasks (if any)")
    function_calling_llm: Optional[Any] = Field(default=None, description="llm to execute func after all agent execution (if any)")
    prompt_file: str = Field(default="", description="path to the prompt json file to be used by the team.")
    process: TaskHandlingProcess = Field(default=TaskHandlingProcess.sequential)

    # callbacks
    before_kickoff_callbacks: List[Callable[[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]]] = Field(
        default_factory=list,
        description="list of callback functions to be executed before the team kickoff. i.e., adjust inputs"
    )
    after_kickoff_callbacks: List[Callable[[TeamOutput], TeamOutput]] = Field(
        default_factory=list,
        description="list of callback functions to be executed after the team kickoff. i.e., store the result in repo"
    )
    task_callback: Optional[Any] = Field(default=None, description="callback to be executed after each task for all agents execution")
    step_callback: Optional[Any] = Field(default=None, description="callback to be executed after each step for all agents execution")

    verbose: bool = Field(default=True)
    cache: bool = Field(default=True)
    memory: bool = Field(default=False, description="whether the team should use memory to store memories of its execution")
    execution_logs: List[Dict[str, Any]] = Field(default=[], description="list of execution logs for tasks")
    usage_metrics: Optional[UsageMetrics] = Field(default=None, description="usage metrics for all the llm executions")


    def __name__(self) -> str:
        return self.name if self.name is not None else self.id.__str__


    @property
    def key(self) -> str:
        source = [str(member.agent.id.__str__) for member in self.members] + [str(task.id.__str__) for task in self.tasks]
        return md5("|".join(source).encode(), usedforsecurity=False).hexdigest()


    @property
    def manager_agent(self) -> Agent:
        manager_agent = [member.agent for member in self.members if member.is_manager == True]
        return manager_agent[0] if len(manager_agent) > 0 else None


    @property
    def manager_task(self) -> Task:
        """
        Aside from the team task, return the task that the `manager_agent` needs to handle.
        The task is set as second priority following to the team tasks.
        """
        task = [member.task for member in self.members if member.is_manager == True]
        return task[0] if len(task) > 0 else None


    @property
    def tasks(self):
        """
        Return all the tasks that the team needs to handle in order of priority:
        1. team tasks,
        2. manager_task,
        3. members' tasks
        """
        sorted_member_tasks = [
            member.task for member in self.members if member.is_manager == True
        ] + [member.task for member in self.members if member.is_manager == False]
        return (
            self.team_tasks + sorted_member_tasks
            if len(self.team_tasks) > 0
            else sorted_member_tasks
        )

    # validators
    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        """Prevent manual setting of the 'id' field by users."""
        if v:
            raise PydanticCustomError("may_not_set_field", "The 'id' field cannot be set by the user.", {})

    # @field_validator("config", mode="before")
    # @classmethod
    # def check_config_type(cls, v: Union[Json, Dict[str, Any]]) -> Union[Json, Dict[str, Any]]:
    #     return json.loads(v) if isinstance(v, Json) else v

    @model_validator(mode="after")
    def check_manager_llm(self):
        """
        Validates that the language model is set when using hierarchical process.
        """

        if self.process == TaskHandlingProcess.hierarchical:
            if self.manager_agent is None:
                raise PydanticCustomError(
                    "missing_manager_llm_or_manager_agent",
                    "Attribute `manager_llm` or `manager_agent` is required when using hierarchical process.",
                    {},
                )

            if (self.manager_agent is not None) and (
                self.members.count(self.manager_agent) > 0
            ):
                raise PydanticCustomError(
                    "manager_agent_in_agents",
                    "Manager agent should not be included in agents list.",
                    {},
                )
        return self


    @model_validator(mode="after")
    def validate_tasks(self):
        """
        Every team member should have a task to handle.
        """
        if self.process == TaskHandlingProcess.sequential:
            for member in self.members:
                if member.task is None:
                    raise PydanticCustomError(
                        "missing_agent_in_task",
                        f"Sequential process error: Agent is missing in the task with the following description: {member.task.description}",
                        {},
                    )
        return self

    @model_validator(mode="after")
    def validate_end_with_at_most_one_async_task(self):
        """
        Validates that the team completes max. one asynchronous task by counting tasks traversed backward
        """

        async_task_count = 0
        for task in reversed(self.tasks):
            if task.async_execution:
                async_task_count += 1
            else:
                break  # stop traversing when a non-async task is found

        if async_task_count > 1:
            raise PydanticCustomError(
                "async_task_count",
                "The team must end with max. one asynchronous task.",
                {},
            )
        return self

    def _get_responsible_agent(self, task: Task) -> Agent:
        res = [member.agent for member in self.members if member.task.id == task.id]
        return None if len(res) == 0 else res[0]

    # setup team planner
    def _handle_team_planning(self):
        team_planner = TeamPlanner(tasks=self.tasks, planner_llm=self.planning_llm)
        result = team_planner._handle_task_planning()

        if result is not None:
            for task in self.tasks:
                task_id = task.id
                task.description += (
                    result[task_id] if hasattr(result, str(task_id)) else result
                )

    # task execution
    def _process_async_tasks(
        self,
        futures: List[Tuple[Task, Future[TaskOutput], int]],
        was_replayed: bool = False,
    ) -> List[TaskOutput]:
        task_outputs: List[TaskOutput] = []
        for future_task, future, task_index in futures:
            task_output = future.result()
            task_outputs.append(task_output)
            self._process_task_result(future_task, task_output)
            self._store_execution_log(
                future_task, task_output, task_index, was_replayed
            )
        return task_outputs

    def _handle_conditional_task(
        self,
        task: ConditionalTask,
        task_outputs: List[TaskOutput],
        futures: List[Tuple[Task, Future[TaskOutput], int]],
        task_index: int,
        was_replayed: bool,
    ) -> Optional[TaskOutput]:
        if futures:
            task_outputs = self._process_async_tasks(futures, was_replayed)
            futures.clear()

        previous_output = task_outputs[task_index - 1] if task_outputs else None
        if previous_output is not None and not task.should_execute(previous_output):
            self._logger.log(
                "debug",
                f"Skipping conditional task: {task.description}",
                color="yellow",
            )
            skipped_task_output = task.get_skipped_task_output()

            if not was_replayed:
                self._store_execution_log(task, skipped_task_output, task_index)
            return skipped_task_output
        return None


    def _create_team_output(self, task_outputs: List[TaskOutput], lead_task_output: TaskOutput = None) -> TeamOutput:
        """
        Take the output of the first task or the lead task output as the team output `raw` value.
        Note that `tasks` are already sorted by the importance.
        """
        if len(task_outputs) < 1:
            raise ValueError("Something went wrong. Kickoff should return only one task output.")

        final_task_output = lead_task_output if lead_task_output is not None else task_outputs[0]
        # final_string_output = final_task_output.raw
        # self._finish_execution(final_string_output)
        token_usage = self._calculate_usage_metrics()

        return TeamOutput(
            team_id=self.id,
            raw=final_task_output.raw,
            json_dict=final_task_output.json_dict,
            pydantic=final_task_output.pydantic,
            task_output_list=task_outputs,
            token_usage=token_usage,
        )


    def _calculate_usage_metrics(self) -> UsageMetrics:
        """
        Calculate and return the usage metrics that consumed by the team.
        """
        total_usage_metrics = UsageMetrics()

        for member in self.members:
            agent = member.agent
            if hasattr(agent, "_token_process"):
                token_sum = agent._token_process.get_summary()
                total_usage_metrics.add_usage_metrics(token_sum)

        if self.manager_agent and hasattr(self.manager_agent, "_token_process"):
            token_sum = self.manager_agent._token_process.get_summary()
            total_usage_metrics.add_usage_metrics(token_sum)

        self.usage_metrics = total_usage_metrics
        return total_usage_metrics


    def _execute_tasks(self, tasks: List[Task], start_index: Optional[int] = 0, was_replayed: bool = False) -> TeamOutput:
        """
        Executes tasks sequentially and returns the final output in TeamOutput class.
        When we have a manager agent, we will start from executing manager agent's tasks.
        Priority
        1. Team tasks > 2. Manager task > 3. Member tasks (in order of index)
        """

        task_outputs: List[TaskOutput] = []
        lead_task_output: TaskOutput = None
        futures: List[Tuple[Task, Future[TaskOutput], int]] = []
        last_sync_output: Optional[TaskOutput] = None

        for task_index, task in enumerate(tasks):
            if start_index is not None and task_index < start_index:
                if task.output:
                    if task.async_execution:
                        task_outputs.append(task.output)
                    else:
                        task_outputs = [task.output]
                        last_sync_output = task.output
                continue

            responsible_agent = self._get_responsible_agent(task)
            if responsible_agent is None:
                responsible_agent = self.manager_agent if self.manager_agent else self.members[0].agent

            # self._prepare_agent_tools(task)
            # self._log_task_start(task, responsible_agent)

            if isinstance(task, ConditionalTask):
                skipped_task_output = self._handle_conditional_task(task, task_outputs, futures, task_index, was_replayed)
                if skipped_task_output:
                    continue

            if task.async_execution:
                context = create_raw_outputs(tasks=[task, ],task_outputs=([last_sync_output,] if last_sync_output else []))
                future = task.execute_async(agent=responsible_agent, context=context,
                                            # tools=responsible_agent.tools
                                            )
                futures.append((task, future, task_index))
            else:
                if futures:
                    task_outputs = self._process_async_tasks(futures, was_replayed)
                    futures.clear()

                context = create_raw_outputs(tasks=[task,], task_outputs=([ last_sync_output,] if last_sync_output else [] ))
                task_output = task.execute_sync(agent=responsible_agent, context=context,
                                                # tools=responsible_agent.tools
                                                )
                if responsible_agent is self.manager_agent:
                    lead_task_output = task_output

                task_outputs.append(task_output)
                # self._process_task_result(task, task_output)
                # self._store_execution_log(task, task_output, task_index, was_replayed)

        # if futures:
        # task_outputs = self._process_async_tasks(futures, was_replayed)
        return self._create_team_output(task_outputs, lead_task_output)


    def kickoff(self, kwargs_before: Optional[Dict[str, str]] = None, kwargs_after: Optional[Dict[str, Any]] = None) -> TeamOutput:
        """
        Kickoff the team:
        0. Plan the team action if we have `team_tasks` using `planning_llm`.
        1. Address `before_kickoff_callbacks` if any.
        2. Handle team members' tasks in accordance with the `process`.
        3. Address `after_kickoff_callbacks` if any.
        """

        metrics: List[UsageMetrics] = []

        if len(self.team_tasks) > 0 or self.planning_llm is not None:
            self._handle_team_planning()

        if kwargs_before is not None:
            for before_callback in self.before_kickoff_callbacks:
                before_callback(**kwargs_before)

        # self._execution_span = self._telemetry.team_execution_span(self, inputs)
        # self._task_output_handler.reset()
        # self._logging_color = "bold_purple"

        # if inputs is not None:
        #     self._inputs = inputs
        # self._interpolate_inputs(inputs)

        for task in self.tasks:
            if not task.callback:
                task.callback = self.task_callback

        # i18n = I18N(prompt_file=self.prompt_file)

        for member in self.members:
            agent = member.agent
            agent.team = self

            # add the team's common callbacks to each agent.
            if not agent.function_calling_llm:
                agent.function_calling_llm = self.function_calling_llm

            # if agent.allow_code_execution:
            #     agent.tools += agent.get_code_execution_tools()

            if not agent.step_callback:
                agent.step_callback = self.step_callback

        if self.process is None:
            self.process = TaskHandlingProcess.sequential

        result = self._execute_tasks(self.tasks)

        for after_callback in self.after_kickoff_callbacks:
            result = after_callback(result, **kwargs_after)

        metrics += [member.agent._token_process.get_summary() for member in self.members]

        self.usage_metrics = UsageMetrics()
        for metric in metrics:
            self.usage_metrics.add_usage_metrics(metric)

        return result
