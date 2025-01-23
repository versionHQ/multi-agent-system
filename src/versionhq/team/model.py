import uuid
import warnings
import json
from enum import Enum
from dotenv import load_dotenv
from concurrent.futures import Future
from hashlib import md5
from typing import Any, Dict, List, TYPE_CHECKING, Callable, Optional, Tuple
from pydantic import UUID4, InstanceOf, Json, BaseModel, Field, PrivateAttr, field_validator, model_validator
from pydantic._internal._generate_schema import GenerateSchema
from pydantic_core import PydanticCustomError, core_schema

from versionhq.agent.model import Agent
from versionhq.task.model import Task, TaskOutput, ConditionalTask, TaskOutputFormat
from versionhq.task.formatter import create_raw_outputs
from versionhq.team.team_planner import TeamPlanner
from versionhq._utils.logger import Logger
from versionhq._utils.usage_metrics import UsageMetrics


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
    json_dict: Dict[str, Any] = Field(default=None, description="`raw` converted to dictionary")
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


class TeamMember(BaseModel):
    agent: Agent | None = Field(default=None, description="store the agent to be a member")
    is_manager: bool = Field(default=False)
    task: Optional[Task] = Field(default=None)

    @property
    def is_idling(self):
        return bool(self.task is None)


class Team(BaseModel):
    """
    A collaborative team of agents that handles complex, multiple tasks.
    We define strategies for task executions and overall workflow.
    """

    __hash__ = object.__hash__
    _execution_span: Any = PrivateAttr()
    _logger: Logger = PrivateAttr()
    _inputs: Optional[Dict[str, Any]] = PrivateAttr(default=None)

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
    step_callback: Optional[Any] = Field(default=None, description="callback to be executed after each step for all agents execution")

    verbose: bool = Field(default=True)
    cache: bool = Field(default=True)
    memory: bool = Field(default=False, description="whether the team should use memory to store memories of its execution")
    execution_logs: List[Dict[str, Any]] = Field(default=[], description="list of execution logs for tasks")
    usage_metrics: Optional[UsageMetrics] = Field(default=None, description="usage metrics for all the llm executions")


    def __name__(self) -> str:
        return self.name if self.name is not None else self.id.__str__


    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        """Prevent manual setting of the 'id' field by users."""
        if v:
            raise PydanticCustomError("may_not_set_field", "The 'id' field cannot be set by the user.", {})


    @model_validator(mode="after")
    def assess_tasks(self):
        """
        Validates if the model recognize all tasks that the team needs to handle.
        """

        if self.tasks:
            if all(task in self.tasks for task in self.team_tasks) == False:
                raise PydanticCustomError("task_validation_error", "`team_tasks` needs to be recognized in the task.", {})

            if len(self.tasks) != len(self.team_tasks) + len([member for member in self.members if member.task is not None]):
                raise PydanticCustomError("task_validation_error", "Some tasks are missing.", {})
        return self


    @model_validator(mode="after")
    def check_manager_llm(self):
        """
        Validates that the language model is set when using hierarchical process.
        """

        if self.process == TaskHandlingProcess.hierarchical:
            if self.managers is None:
                raise PydanticCustomError(
                    "missing_manager_llm_or_manager",
                    "Attribute `manager_llm` or `manager` is required when using hierarchical process.",
                    {},
                )

            if self.managers and (self.manager_tasks is None or self.team_tasks is None):
                raise PydanticCustomError("missing_manager_task", "manager needs to have at least one manager task or team task.", {})

        return self


    @model_validator(mode="after")
    def validate_tasks(self):
        """
        Sequential task processing without any team tasks require a task-agent pairing.
        """

        if self.process == TaskHandlingProcess.sequential and self.team_tasks is None:
            for member in self.members:
                if member.task is None:
                    raise PydanticCustomError("missing_agent_in_task", "Sequential process error: Agent is missing the task", {})
        return self

    @model_validator(mode="after")
    def validate_end_with_at_most_one_async_task(self):
        """
        Validates that the team completes max. one asynchronous task by counting tasks traversed backward
        """

        async_task_count = 0
        for task in reversed(self.tasks):
            if not task:
                break
            elif task.async_execution:
                async_task_count += 1
            else:
                break

        if async_task_count > 1:
            raise PydanticCustomError("async_task_count", "The team must end with max. one asynchronous task.", {})
        return self


    def _get_responsible_agent(self, task: Task) -> Agent:
        if task is None:
            return None
        else:
            res = [member.agent for member in self.members if member.task and member.task.id == task.id]
            return None if len(res) == 0 else res[0]


    def _handle_team_planning(self) -> None:
        """
        Form a team considering agents and tasks given, and update `self.members` field:
            1. Idling managers to take the team tasks.
            2. Idling members to take the remaining tasks starting from the team tasks to member tasks.
            3. Create agents to handle the rest tasks.
        """

        team_planner = TeamPlanner(tasks=self.tasks, planner_llm=self.planning_llm)
        idling_managers: List[TeamMember] = [member for member in self.members if member.is_idling and member.is_manager is True]
        idling_members: List[TeamMember] =  [member for member in self.members if member.is_idling and member.is_manager is False]
        unassigned_tasks: List[Task] = self.member_tasks_without_agent
        new_team_members: List[TeamMember] = []

        if self.team_tasks:
            candidates = idling_managers + idling_members
            if candidates:
                i = 0
                while i < len(candidates):
                    if self.team_tasks[i]:
                        candidates[i].task = self.team_tasks[i]
                        i += 1

                if len(self.team_tasks) > i:
                    for item in self.team_tasks[i:]:
                        if item not in unassigned_tasks:
                            unassigned_tasks = [item, ] + unassigned_tasks

            else:
                for item in self.team_tasks:
                    if item not in unassigned_tasks:
                        unassigned_tasks = [item, ] + unassigned_tasks

        if unassigned_tasks:
            new_team_members = team_planner._handle_assign_agents(unassigned_tasks=unassigned_tasks)

        if new_team_members:
            self.members += new_team_members


    # task execution
    def _process_async_tasks(
        self, futures: List[Tuple[Task, Future[TaskOutput], int]], was_replayed: bool = False
    ) -> List[TaskOutput]:
        """
        When we have `Future` tasks, updated task outputs and task execution logs accordingly.
        """

        task_outputs: List[TaskOutput] = []

        for future_task, future, task_index in futures:
            task_output = future.result()
            task_outputs.append(task_output)
            future_task._store_execution_log(task_index, was_replayed)

        return task_outputs


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

        if self.managers:
            for manager in self.managers:
                if hasattr(manager.agent, "_token_process"):
                    token_sum = manager.agent._token_process.get_summary()
                    total_usage_metrics.add_usage_metrics(token_sum)

        self.usage_metrics = total_usage_metrics
        return total_usage_metrics


    def _execute_tasks(self, tasks: List[Task], start_index: Optional[int] = 0, was_replayed: bool = False) -> TeamOutput:
        """
        Executes tasks sequentially and returns the final output in TeamOutput class.
        When we have a manager agent, we will start from executing manager agent's tasks.
        Priority:
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
                self._handle_team_planning()

            if isinstance(task, ConditionalTask):
                skipped_task_output = task._handle_conditional_task(task_outputs, futures, task_index, was_replayed)
                if skipped_task_output:
                    continue

            # self._log_task_start(task, responsible_agent)

            if task.async_execution:
                context = create_raw_outputs(tasks=[task, ], task_outputs=([last_sync_output,] if last_sync_output else []))
                future = task.execute_async(agent=responsible_agent, context=context)
                futures.append((task, future, task_index))
            else:
                context = create_raw_outputs(tasks=[task,], task_outputs=([last_sync_output,] if last_sync_output else [] ))
                task_output = task.execute_sync(agent=responsible_agent, context=context)
                if self.managers and responsible_agent in [manager.agent for manager in self.managers]:
                    lead_task_output = task_output

                task_outputs.append(task_output)
                # self._process_task_result(task, task_output)
                task._store_execution_log(task_index, was_replayed, self._inputs)


        if futures:
            task_outputs = self._process_async_tasks(futures, was_replayed)

        return self._create_team_output(task_outputs, lead_task_output)


    def kickoff(self, kwargs_before: Optional[Dict[str, str]] = None, kwargs_after: Optional[Dict[str, Any]] = None) -> TeamOutput:
        """
        Kickoff the team:
        0. Assign an agent to a task - using conditions (manager prioritizes team_tasks) and planning_llm.
        1. Address `before_kickoff_callbacks` if any.
        2. Handle team members' tasks in accordance with the process.
        3. Address `after_kickoff_callbacks` if any.
        """

        metrics: List[UsageMetrics] = []

        if self.team_tasks or self.member_tasks_without_agent:
            self._handle_team_planning()

        if kwargs_before is not None:
            for before_callback in self.before_kickoff_callbacks:
                before_callback(**kwargs_before)

        # self._execution_span = self._telemetry.team_execution_span(self, inputs)
        # self._task_output_handler.reset()
        # self._logging_color = "bold_purple"


        # i18n = I18N(prompt_file=self.prompt_file)

        for member in self.members:
            agent = member.agent
            agent.team = self

            if not agent.function_calling_llm and self.function_calling_llm:
                agent.function_calling_llm = self.function_calling_llm

            if self.step_callback:
                agent.callbacks.append(self.step_callback)

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


    @property
    def key(self) -> str:
        source = [str(member.agent.id.__str__) for member in self.members] + [str(task.id.__str__) for task in self.tasks]
        return md5("|".join(source).encode(), usedforsecurity=False).hexdigest()


    @property
    def managers(self) -> List[TeamMember] | None:
        managers = [member for member in self.members if member.is_manager == True]
        return managers if len(managers) > 0 else None


    @property
    def manager_tasks(self) -> List[Task] | None:
        """
        Tasks (incl. team tasks) handled by managers in the team.
        """
        if self.managers:
            tasks = [manager.task for manager in self.managers if manager.task is not None]
            return tasks if len(tasks) > 0 else None

        return None


    @property
    def tasks(self):
        """
        Return all the tasks that the team needs to handle in order of priority:
        1. team tasks, -> assigned to the member
        2. manager_task,
        3. members' tasks
        """

        team_tasks = self.team_tasks
        manager_tasks = [member.task for member in self.members if member.is_manager == True and member.task is not None and member.task not in team_tasks]
        member_tasks = [member.task for member in self.members if member.is_manager == False and member.task is not None and member.task not in team_tasks]

        return team_tasks + manager_tasks + member_tasks


    @property
    def member_tasks_without_agent(self) -> List[Task] | None:
        if self.members:
            return [member.task for member in self.members if member.agent is None]

        return None
