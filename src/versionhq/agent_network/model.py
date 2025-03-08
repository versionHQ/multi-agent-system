import uuid
import warnings
from enum import Enum
from concurrent.futures import Future
from hashlib import md5
from typing import Any, Dict, List, Callable, Optional, Tuple
from typing_extensions import Self

from pydantic import UUID4, BaseModel, Field, PrivateAttr, field_validator, model_validator
from pydantic._internal._generate_schema import GenerateSchema
from pydantic_core import PydanticCustomError, core_schema


from versionhq.agent.model import Agent
from versionhq.task.model import Task, TaskOutput, TaskExecutionType, ResponseField
from versionhq.task_graph.model import TaskGraph, Node, Edge, TaskStatus, DependencyType, Condition
from versionhq._utils.logger import Logger
# from versionhq.recording.usage_metrics import UsageMetrics


initial_match_type = GenerateSchema.match_type

def match_type(self, obj):
    if getattr(obj, "__name__", None) == "datetime":
        return core_schema.datetime_schema()
    return initial_match_type(self, obj)


GenerateSchema.match_type = match_type
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


class Formation(str, Enum):
    SOLO = 1
    SUPERVISING = 2
    SQUAD = 3
    RANDOM = 4
    HYBRID = 10


class TaskHandlingProcess(str, Enum):
    """
    A class representing task handling processes to tackle multiple tasks.
    When the agent network has multiple tasks that connect with edges, follow the edge conditions.
    """
    HIERARCHY = 1
    SEQUENTIAL = 2
    CONSENSUAL = 3 # either from manager class agents from diff network or human. need to define a trigger


class Member(BaseModel):
    """
    A class to store a member (agent) in the network, with its tasks and memory/knowledge share settings.
    """
    agent: Agent | None = Field(default=None)
    is_manager: bool = Field(default=False)
    can_share_knowledge: bool = Field(default=True, description="whether to share the agent's knowledge in the network")
    can_share_memory: bool = Field(default=True, description="whether to share the agent's memory in the network")
    tasks: Optional[List[Task]] = Field(default_factory=list, description="tasks explicitly assigned to the agent")

    @property
    def is_idling(self):
        return bool(self.tasks)


class AgentNetwork(BaseModel):
    """A Pydantic class to store an agent network with agent members and tasks."""

    __hash__ = object.__hash__
    _execution_span: Any = PrivateAttr()
    _inputs: Optional[Dict[str, Any]] = PrivateAttr(default=None)

    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    name: Optional[str] = Field(default=None)
    members: List[Member] = Field(default_factory=list)
    formation: Optional[Formation] = Field(default=None)
    should_reform: bool = Field(default=False, description="True if task exe. failed or eval scores below threshold")

    network_tasks: Optional[List[Task]] = Field(default_factory=list, description="tasks without dedicated agents - network's common tasks")

    # task execution rules
    prompt_file: str = Field(default="", description="absolute file path to the prompt file that stores jsonified prompts")
    process: TaskHandlingProcess = Field(default=TaskHandlingProcess.SEQUENTIAL)
    consent_trigger: Optional[Callable | Condition] = Field(default=None, description="returns bool")

    # callbacks
    pre_launch_callbacks: List[Callable[..., Any]]= Field(default_factory=list, description="list of callback funcs called before the network launch")
    post_launch_callbacks: List[Callable[..., Any]] = Field(default_factory=list, description="list of callback funcs called after the network launch")
    step_callback: Optional[Any] = Field(default=None, description="callback to be executed after each step of all agents execution")

    cache: bool = Field(default=True)
    execution_logs: List[Dict[str, Any]] = Field(default_factory=list, description="list of execution logs of the tasks handled by members")
    # usage_metrics: Optional[UsageMetrics] = Field(default=None, description="usage metrics for all the llm executions")


    def __name__(self) -> str:
        return self.name if self.name is not None else self.id.__str__


    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        """Prevent manual setting of the 'id' field by users."""
        if v:
            raise PydanticCustomError("may_not_set_field", "The 'id' field cannot be set by the user.", {})


    @model_validator(mode="after")
    def validate_process(self) -> Self:
        if self.process == TaskHandlingProcess.CONSENSUAL and not self.consent_trigger:
            Logger().log(level="error", message="Need to define the consent trigger function that returns bool", color="red")
            raise PydanticCustomError("invalid_process", "Need to define the consent trigger function that returns bool", {})


        if self.consent_trigger and isinstance(self.consent_trigger, Callable):
            self.consent_trigger = Condition(methods={"0": self.consent_trigger})
        return self


    @model_validator(mode="after")
    def set_up_network_for_members(self) -> Self:
        if self.members:
            for member in self.members:
                member.agent.networks.append(self)

        return self


    @model_validator(mode="after")
    def assess_tasks(self):
        """
        Validates if the model recognize all tasks that the network needs to handle.
        """

        if self.tasks:
            if all(task in self.tasks for task in self.network_tasks) == False:
                raise PydanticCustomError("task_validation_error", "`network_tasks` needs to be recognized in the task.", {})


            num_member_tasks = 0
            for member in self.members:
                num_member_tasks += len(member.tasks)

            # if len(self.tasks) != len(self.network_tasks) + num_member_tasks:
            #     raise PydanticCustomError("task_validation_error", "Some tasks are missing.", {})
        return self


    @model_validator(mode="after")
    def check_manager(self):
        """
        Check if the agent network has a manager
        """
        if self.process == TaskHandlingProcess.HIERARCHY or self.formation == Formation.SUPERVISING:
            if not self.managers:
                Logger().log(level="error", message="The process or formation created needs at least 1 manager agent.", color="red")
                raise PydanticCustomError("missing_manager", "`manager` is required when using hierarchical process.", {})

        ## comment out for the formation flexibilities
        # if self.managers and (self.manager_tasks is None or self.network_tasks is None):
        #     Logger().log(level="error", message="The manager is idling. At least 1 task needs to be assigned to the manager.", color="red")
        #     raise PydanticCustomError("missing_manager_task", "manager needs to have at least one manager task or network task.", {})

        return self


    @model_validator(mode="after")
    def validate_task_member_paring(self):
        """
        Sequential task processing without any network_tasks require a task-agent pairing.
        """
        if self.process == TaskHandlingProcess.SEQUENTIAL and self.network_tasks is None:
            for task in self.tasks:
                if not [member.task == task for member in self.members]:
                    Logger().log(level="error", message=f"The following task needs a dedicated agent to be assinged: {task.description}", color="red")
                    raise PydanticCustomError("missing_agent_in_task", "Sequential process error: Agent is missing the task", {})
        return self


    @model_validator(mode="after")
    def validate_end_with_at_most_one_async_task(self):
        """
        Validates that the agent network completes max. one asynchronous task by counting tasks traversed backward
        """

        async_task_count = 0
        for task in reversed(self.tasks):
            if not task:
                break
            elif task.execution_type == TaskExecutionType.ASYNC:
                async_task_count += 1
            else:
                break

        if async_task_count > 1:
            raise PydanticCustomError("async_task_count", "The agent network must end with at maximum one asynchronous task.", {})
        return self


    def _generate_agents(self, unassigned_tasks: List[Task]) -> List[Member]:
        """Generates agents as members in the network."""

        from versionhq.agent.inhouse_agents import vhq_agent_creator

        new_member_list: List[Member] = []

        for unassgined_task in unassigned_tasks:
            task = Task(
                description=f"Based on the following task summary, draft an agent's role and goal in concise manner. Task summary: {unassgined_task.summary}",
                response_fields=[
                    ResponseField(title="goal", data_type=str, required=True),
                    ResponseField(title="role", data_type=str, required=True),
                ],
            )
            res = task.execute(agent=vhq_agent_creator)
            agent = Agent(
                role=res.json_dict["role"] if "role" in res.json_dict else res.raw,
                goal=res.json_dict["goal"] if "goal" in res.json_dict else task.description
            )
            if agent.id:
                member = Member(agent=agent, tasks=[unassgined_task], is_manager=False)
                new_member_list.append(member)

        return new_member_list


    def _assign_tasks(self) -> None:
        """Assigns tasks to member agents."""

        idling_managers: List[Member] = [member for member in self.members if member.is_idling and member.is_manager == True]
        idling_members: List[Member] =  [member for member in self.members if member.is_idling and member.is_manager == False]
        unassigned_tasks: List[Task] = self.network_tasks + self.unassigned_member_tasks if self.network_tasks else self.unassigned_member_tasks
        new_members: List[Member] = []

        if not unassigned_tasks:
            return

        else:
            if idling_managers:
                idling_managers[0].tasks.extend(unassigned_tasks)

            elif idling_members:
                idling_members[0].tasks.extend(unassigned_tasks)

            else:
                new_members = self._generate_agents(unassigned_tasks=unassigned_tasks)
                if new_members:
                    self.members += new_members


    def _get_responsible_agent(self, task: Task) -> Agent | None:
        if not task:
            return None

        self._assign_tasks()
        for member in self.members:
            if member.tasks and [item for item in member.tasks if item.id == task.id]:
                return member.agent


    def _execute_tasks(self, tasks: List[Task], start_index: Optional[int] = None) -> Tuple[TaskOutput, TaskGraph]:
        """Executes tasks and returns TaskOutput object as concl or latest response in the network."""
        res, task_graph = None, None

        if len(tasks) == 1:
            task = self.tasks[0]
            responsible_agent = self._get_responsible_agent(task=task)
            res = task.execute(agent=responsible_agent)
            node = Node(task=task)
            task_graph = TaskGraph(nodes={ node.identifier: node, }, concl=res if res else None)
            return res, task_graph

        nodes = [
            Node(
                task=task,
                assigned_to=self._get_responsible_agent(task=task),
                status=TaskStatus.NOT_STARTED if not start_index or i >= start_index else TaskStatus.COMPLETED,
            ) for i, task in enumerate(tasks)
        ]
        task_graph = TaskGraph(nodes={node.identifier: node for node in nodes})

        for i in range(0, len(nodes) - 1):
            condition = self.consent_trigger if isinstance(self.consent_trigger, Condition) else Condition(methods={"0": self.consent_trigger }) if self.consent_trigger else None
            task_graph.add_edge(
                source=nodes[i].identifier,
                target=nodes[i+1].identifier,
                edge=Edge(
                    weight=3 if nodes[i].task in self.manager_tasks else 1,
                    dependency_type=DependencyType.FINISH_TO_START if self.process == TaskHandlingProcess.HIERARCHY else DependencyType.START_TO_START,
                    required=bool(self.process == TaskHandlingProcess.CONSENSUAL),
                    condition=condition,
                    data_transfer=bool(self.process == TaskHandlingProcess.HIERARCHY),
                )
            )

        if start_index is not None:
            res, _ = task_graph.activate(target=nodes[start_index].indentifier)

        else:
            res, _ = task_graph.activate()

        if not res:
            Logger().log(level="error", message="Missing task outputs.", color="red")
            raise ValueError("Missing task outputs")

        return res, task_graph


    def launch(self, kwargs_pre: Optional[Dict[str, str]] = None, kwargs_post: Optional[Dict[str, Any]] = None, start_index: int = None) -> Tuple[TaskOutput, TaskGraph]:
        """Launches agent network by executing tasks in the network and recording the outputs"""

        self._assign_tasks()

        if kwargs_pre:
            for func in self.pre_launch_callbacks: #! REFINEME -  signature check
                func(**kwargs_pre)

        for member in self.members:
            agent = member.agent

            if not agent.networks:
                agent.networks.append(self)

            if self.step_callback:
                agent.callbacks.append(self.step_callback)

        if self.process is None:
            self.process = TaskHandlingProcess.SEQUENTIAL

        result, tg = self._execute_tasks(self.tasks, start_index=start_index)
        callback_output = None

        for func in self.post_launch_callbacks:
            callback_output = func(result, **kwargs_post)

        if callback_output:
            match result:
                case TaskOutput():
                    result.callback_output = callback_output

                case _:
                    pass

        return result, tg


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
        Tasks (incl. network tasks) handled by managers in the agent network.
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
        Return all the tasks that the agent network needs to handle in order of priority:
        1. network_tasks, -> assigned to the member
        2. manager_task,
        3. members' tasks
        """

        network_tasks = self.network_tasks
        manager_tasks = self.manager_tasks
        member_tasks = []

        for member in self.members:
            if member.is_manager == False and member.tasks:
                a = [item for item in member.tasks if item not in network_tasks and item not in manager_tasks]
                member_tasks += a

        return network_tasks + manager_tasks + member_tasks


    @property
    def unassigned_member_tasks(self) -> List[Task]:
        res = list()

        if self.members:
            for member in self.members:
                if member.agent is None and member.tasks:
                    res.extend(member.tasks)

        return res
