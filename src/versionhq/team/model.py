import uuid
import warnings
from enum import Enum
from dotenv import load_dotenv
from concurrent.futures import Future
from hashlib import md5
from typing import Any, Dict, List, Callable, Optional, Tuple
from pydantic import UUID4, BaseModel, Field, PrivateAttr, field_validator, model_validator
from pydantic._internal._generate_schema import GenerateSchema
from pydantic_core import PydanticCustomError, core_schema

from versionhq.agent.model import Agent
from versionhq.task.model import Task, TaskOutput, ConditionalTask
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



class Formation(str, Enum):
    UNDEFINED = 0
    SOLO = 1
    SUPERVISING = 2
    NETWORK = 3
    RANDOM = 4
    HYBRID = 10


class TaskHandlingProcess(str, Enum):
    """
    Class representing the different processes that can be used to tackle multiple tasks.
    """
    sequential = "sequential"
    hierarchical = "hierarchical"
    consensual = "consensual"


class TeamOutput(TaskOutput):
    """
    A class to store output from the team, inherited from TaskOutput class.
    """

    team_id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True, description="store the team ID that generate the TeamOutput")
    task_description: str = Field(default=None, description="store initial request (task description) from the client")
    task_outputs: list[TaskOutput] = Field(default=list, description="store outputs of all tasks that the team has executed")
    token_usage: UsageMetrics = Field(default=dict, description="processed token summary")


    def return_all_task_outputs(self) -> List[Dict[str, Any]]:
        res = [output.json_dict for output in self.task_outputs]
        return res


    def __str__(self):
        return (str(self.pydantic) if self.pydantic else str(self.json_dict) if self.json_dict else self.raw)


    def __getitem__(self, key):
        if self.pydantic and hasattr(self.pydantic, key):
            return getattr(self.pydantic, key)
        elif self.json_dict and key in self.json_dict:
            return self.json_dict[key]
        else:
            raise KeyError(f"Key '{key}' not found in the team output.")



class Member(BaseModel):
    """
    A class to store a member in the network and connect the agent as a member with tasks and sharable settings.
    """
    agent: Agent | None = Field(default=None)
    is_manager: bool = Field(default=False)
    can_share_knowledge: bool = Field(default=True, description="whether to share the agent's knowledge in the team")
    can_share_memory: bool = Field(default=True, description="whether to share the agent's memory in the team")
    tasks: Optional[List[Task]] = Field(default_factory=list, description="tasks explicitly assigned to the agent")

    @property
    def is_idling(self):
        return bool(self.tasks)


class Team(BaseModel):
    """
    A class to store agent network that shares knowledge, memory and tools among the members.
    """

    __hash__ = object.__hash__
    _execution_span: Any = PrivateAttr()
    _logger: Logger = PrivateAttr(default_factory=lambda: Logger(verbose=True))
    _inputs: Optional[Dict[str, Any]] = PrivateAttr(default=None)

    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    name: Optional[str] = Field(default=None)
    members: List[Member] = Field(default_factory=list)
    formation: Optional[Formation] = Field(default=None)
    should_reform: bool = Field(default=False, description="True if task exe. failed or eval scores below threshold")

    # formation planning
    planner_llm: Optional[Any] = Field(default=None, description="llm to generate and evaluate formation")
    team_tasks: Optional[List[Task]] = Field(default_factory=list, description="tasks without dedicated agents to handle")

    # task execution rules
    prompt_file: str = Field(default="", description="absolute path to the prompt json file")
    process: TaskHandlingProcess = Field(default=TaskHandlingProcess.sequential)

    # callbacks
    pre_launch_callbacks: List[Callable[[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]]] = Field(
        default_factory=list,
        description="list of callback functions to be executed before the team launch. i.e., adjust inputs"
    )
    post_launch_callbacks: List[Callable[[TeamOutput], TeamOutput]] = Field(
        default_factory=list,
        description="list of callback functions to be executed after the team launch. i.e., store the result in repo"
    )
    step_callback: Optional[Any] = Field(default=None, description="callback to be executed after each step for all agents execution")

    cache: bool = Field(default=True)
    execution_logs: List[Dict[str, Any]] = Field(default_factory=list, description="list of execution logs of the tasks handled by members")
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


            num_member_tasks = 0
            for member in self.members:
                num_member_tasks += len(member.tasks)

            # if len(self.tasks) != len(self.team_tasks) + num_member_tasks:
            #     raise PydanticCustomError("task_validation_error", "Some tasks are missing.", {})
        return self


    @model_validator(mode="after")
    def check_manager_llm(self):
        """
        Check if the team has a manager
        """

        if self.process == TaskHandlingProcess.hierarchical or self.formation == Formation.SUPERVISING:
            if not self.managers:
                self._logger.log(level="error", message="The process or formation created needs at least 1 manager agent.", color="red")
                raise PydanticCustomError(
                    "missing_manager_llm_or_manager","Attribute `manager_llm` or `manager` is required when using hierarchical process.", {})

            ## comment out for the formation flexibilities
            # if self.managers and (self.manager_tasks is None or self.team_tasks is None):
            #     self._logger.log(level="error", message="The manager is idling. At least 1 task needs to be assigned to the manager.", color="red")
            #     raise PydanticCustomError("missing_manager_task", "manager needs to have at least one manager task or team task.", {})

        return self


    @model_validator(mode="after")
    def validate_task_member_paring(self):
        """
        Sequential task processing without any team tasks require a task-agent pairing.
        """
        if self.process == TaskHandlingProcess.sequential and self.team_tasks is None:
            for task in self.tasks:
                if not [member.task == task for member in self.members]:
                    self._logger.log(level="error", message=f"The following task needs a dedicated agent to be assinged: {task.description}", color="red")
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


    def _get_responsible_agent(self, task: Task) -> Agent | None:
        if task is None:
            return None
        else:
            for member in self.members:
                if member.tasks and [item for item in member.tasks if item.id == task.id]:
                    return member.agent

            return None


    def _handle_agent_formation(self) -> None:
        """
        Form a team considering agents and tasks given, and update `self.members` field:
            1. Idling managers to take the team tasks.
            2. Idling members to take the remaining tasks starting from the team tasks to member tasks.
            3. Create agents to handle the rest tasks.
        """

        team_planner = TeamPlanner(tasks=self.tasks, planner_llm=self.planner_llm)
        idling_managers: List[Member] = [member for member in self.members if member.is_idling and member.is_manager is True]
        idling_members: List[Member] =  [member for member in self.members if member.is_idling and member.is_manager is False]
        unassigned_tasks: List[Task] = self.member_tasks_without_agent
        new_team_members: List[Member] = []

        if self.team_tasks:
            candidates = idling_managers + idling_members
            if candidates:
                i = 0
                while i < len(candidates):
                    if len(self.team_tasks) < i and self.team_tasks[i]:
                        candidates[i].tasks.append(self.team_tasks[i])
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
    def _process_async_tasks(self, futures: List[Tuple[Task, Future[TaskOutput], int]], was_replayed: bool = False) -> List[TaskOutput]:
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

        if not task_outputs:
            self._logger.log(level="error", message="Missing task outcomes. Failed to launch the task.", color="red")
            raise ValueError("Failed to launch tasks")

        final_task_output = lead_task_output if lead_task_output is not None else task_outputs[0] #! REFINEME
        # final_string_output = final_task_output.raw
        # self._finish_execution(final_string_output)
        token_usage = self._calculate_usage_metrics()

        return TeamOutput(
            team_id=self.id,
            raw=final_task_output.raw,
            json_dict=final_task_output.json_dict,
            pydantic=final_task_output.pydantic,
            task_outputs=task_outputs,
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
                self._handle_agent_formation()

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


    def launch(self, kwargs_pre: Optional[Dict[str, str]] = None, kwargs_post: Optional[Dict[str, Any]] = None) -> TeamOutput:
        """
        Confirm and launch the formation - execute tasks and record outputs.
        0. Assign an agent to a task - using conditions (manager prioritizes team_tasks) and planner_llm.
        1. Address `pre_launch_callbacks` if any.
        2. Handle team members' tasks in accordance with the process.
        3. Address `post_launch_callbacks` if any.
        """

        metrics: List[UsageMetrics] = []

        if self.team_tasks or self.member_tasks_without_agent:
            self._handle_agent_formation()

        if kwargs_pre is not None:
            for func in self.pre_launch_callbacks:
                func(**kwargs_pre)

        # self._execution_span = self._telemetry.team_execution_span(self, inputs)
        # self._task_output_handler.reset()
        # self._logging_color = "bold_purple"
        # i18n = I18N(prompt_file=self.prompt_file)

        for member in self.members:
            agent = member.agent
            agent.team = self

            if self.step_callback:
                agent.callbacks.append(self.step_callback)

        if self.process is None:
            self.process = TaskHandlingProcess.sequential

        result = self._execute_tasks(self.tasks)

        for func in self.post_launch_callbacks:
            result = func(result, **kwargs_post)

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
    def managers(self) -> List[Member] | None:
        managers = [member for member in self.members if member.is_manager == True]
        return managers if len(managers) > 0 else None


    @property
    def manager_tasks(self) -> List[Task]:
        """
        Tasks (incl. team tasks) handled by managers in the team.
        """
        res = list()

        if self.managers:
            for manager in self.managers:
                if manager.tasks:
                    res.extend(manager.tasks)

        return res


    @property
    def tasks(self) -> List[Task]:
        """
        Return all the tasks that the team needs to handle in order of priority:
        1. team tasks, -> assigned to the member
        2. manager_task,
        3. members' tasks
        """

        team_tasks = self.team_tasks
        manager_tasks = self.manager_tasks
        member_tasks = []

        for member in self.members:
            if member.is_manager == False and member.tasks:
                a = [item for item in member.tasks if item not in team_tasks and item not in manager_tasks]
                member_tasks += a

        return team_tasks + manager_tasks + member_tasks


    @property
    def member_tasks_without_agent(self) -> List[Task]:
        res = list()

        if self.members:
            for member in self.members:
                if member.agent is None and member.tasks:
                    res.extend(member.tasks)

        return res
