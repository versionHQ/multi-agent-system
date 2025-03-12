from typing import List, Optional, Dict
from typing_extensions import Self

from pydantic import InstanceOf, Field

from versionhq.agent.model import Agent
from versionhq.task.model import Task
from versionhq.task_graph.model import TaskGraph, Node, DependencyType, ReformTriggerEvent
from versionhq._prompt.model import Prompt
from versionhq._prompt.constants import REFLECT, INTEGRATE, parameter_sets


class PromptFeedbackGraph(TaskGraph):
    """A Pydantic class to handle auto prompt feedback cycle."""

    _times_iteration: int = 0
    user_prompts: Optional[Dict[str, str]] = Field(default_factory=dict) # { "0": "...", "1": "..."}
    dev_prompts: Optional[Dict[str, str]] = Field(default_factory=dict)
    prompts: Optional[Dict[str, InstanceOf[Prompt]]] = Field(default_factory=dict)


    def __init__(self, prompt: InstanceOf[Prompt] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if prompt:
            user_prompt, dev_prompt, _ = prompt.format_core()
            self.prompts = { self.key: prompt }
            self.user_prompts = { self.key: user_prompt }
            self.dev_prompts = { self.key: dev_prompt }


    def _fetch_latest_prompt(self) -> InstanceOf[Prompt] | None:
        return self.prompts[self.key] if self.key in self.prompts else None


    def _generate_agents(self) -> List[Agent] | None:
        agents = []
        prompt = self._fetch_latest_prompt()

        if not prompt:
            return None

        agent = prompt.agent
        agent_params = agent.model_dump(exclude={"id", "llm", "llm_config", "self_learning"})
        for params in parameter_sets:
            agent = Agent(**agent_params, llm=agent.llm.model, llm_config={**params}, self_learning=True)
            agents.append(agent)
        return agents


    def _reflect(self, original_response: str) -> Task:
        description = REFLECT.format(original_prompt=self.original_prompt, original_response=original_response)
        return Task(description=description)


    def set_up_graph(self, **attributes) -> Self:
        """Sets up a TaskGraph object with nodes and edges."""

        prompt = self._fetch_latest_prompt()
        base_task = prompt.task if prompt else None
        base_agent = prompt.agent if prompt else None

        if not base_task or not base_agent:
            return None

        agents = self._generate_agents()
        if not agents:
            return None

        self.concl_template = base_task.pydantic_output if base_task.pydantic_output else base_task.response_fields if base_task.response_fields else None
        base_agent.callbacks.append(self._reflect)
        init_node = Node(task=base_task, assigned_to=base_agent)
        self.add_node(init_node)

        final_task = Task(description=INTEGRATE.format(original_prompt=self.original_prompt, responses=""))
        final_node = Node(task=final_task, agent=base_agent)
        self.add_node(node=final_node)

        for agent in agents:
            node = Node(task=base_task, assigned_to=agent)
            self.add_node(node=node)
            self.add_dependency(source=init_node.identifier, target=node.identifier, dependency_type=DependencyType.FINISH_TO_START, required=True)
            self.add_dependency(source=node.identifier, target=final_node.identifier, dependency_type=DependencyType.FINISH_TO_START, required=True)

        if attributes:
            for k, v in attributes.items():
                if hasattr(self, k):
                    setattr(self, k, v)

        return self

    @property
    def index(self) -> str:
        """Returns an index to add new item."""
        return str(len([k for k in self.user_prompts.keys()]))

    @property
    def original_prompt(self) -> str:
        return str(self.user_prompts["0"]) + str(self.dev_prompts["0"])

    @property
    def key(self):
        return str(self._times_iteration)
