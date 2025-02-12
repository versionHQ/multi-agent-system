from typing import List, Type
from enum import Enum

from pydantic import BaseModel

from versionhq.task.model import Task
from versionhq.agent.model import Agent
from versionhq.agent_network.model import AgentNetwork, Member, Formation
from versionhq.agent.inhouse_agents  import vhq_formation_planner
from versionhq._utils import Logger


def form_agent_network(
        task: str,
        expected_outcome: str,
        agents: List[Agent] = None,
        context: str = None,
        formation: Type[Formation] = None
    ) -> AgentNetwork | None:
    """
    Make a formation of agents from the given task description, expected outcome, agents (optional), and context (optional).
    """

    if not task:
        Logger(verbose=True).log(level="error", message="Missing task description.", color="red")
        return None

    if not expected_outcome:
        Logger(verbose=True).log(level="error", message="Missing expected outcome.", color="red")
        return None

    if formation:
        try:
            match formation:
                case Formation():
                    if formation == Formation.UNDEFINED:
                        formation = None
                    else:
                        pass

                case str():
                    matched = [item for item in Formation._member_names_ if item == formation.upper()]
                    if matched:
                        formation = getattr(Formation, matched[0])
                    else:
                        # Formation._generate_next_value_(name=f"CUSTOM_{formation.upper()}", start=100, count=6, last_values=Formation.HYBRID.name)
                        Logger(verbose=True).log(level="warning", message=f"The formation {formation} is invalid. We'll recreate a valid formation.", color="yellow")
                        formation = None

                case int() | float():
                    formation = Formation(int(formation))

                case _:
                    Logger(verbose=True).log(level="warning", message=f"The formation {formation} is invalid. We'll recreate a valid formation.", color="yellow")
                    formation = None

        except Exception as e:
            Logger(verbose=True).log(level="warning", message=f"The formation {formation} is invalid: {str(e)}. We'll recreate a formation.", color="yellow")
            formation = None

    try:
        prompt_formation = formation.name if formation and isinstance(formation, Formation) else f"Select the best formation to effectively execute the tasks from the given Enum sets: {str(Formation.__dict__)}."
        class Outcome(BaseModel):
            formation: Enum
            agent_roles: list[str]
            task_descriptions: list[str]
            leader_agent: str

        vhq_task = Task(
            description=f"""
    Create a team of specialized agents designed to automate the following task and deliver the expected outcome. Consider the necessary roles for each agent with a clear task description. If you think we neeed a leader to handle the automation, return a leader_agent role as well, but if not, leave the a leader_agent role blank. When you have a leader_agent, the formation must be SUPERVISING or HYBRID.
    Task: {str(task)}
    Expected outcome: {str(expected_outcome)}
    Formation: {prompt_formation}
            """,
            pydantic_output=Outcome
        )

        if agents:
            vhq_task.description += "Consider adding following agents in the formation: " + ", ".join([agent.role for agent in agents if isinstance(agent, Agent)])

        res = vhq_task.execute(agent=vhq_formation_planner, context=context)
        _formation = Formation.SUPERVISING

        if res.pydantic:
            formation_keys = [k for k, v in Formation._member_map_.items() if k == res.pydantic.formation.upper()]

            if formation_keys:
                _formation = Formation[formation_keys[0]]

            created_agents = [Agent(role=item, goal=item) for item in res.pydantic.agent_roles]
            created_tasks = [Task(description=item) for item in res.pydantic.task_descriptions]

            network_tasks = []
            members = []
            leader = str(res.pydantic.leader_agent)

            for i in range(len(created_agents)):
                is_manager = bool(created_agents[i].role.lower() == leader.lower())
                member = Member(agent=created_agents[i], is_manager=is_manager)

                if len(created_tasks) >= i and created_tasks[i]:
                    member.tasks.append(created_tasks[i])

                members.append(member)


            if len(created_agents) < len(created_tasks):
                network_tasks.extend(created_tasks[len(created_agents):len(created_tasks)])

            members.sort(key=lambda x: x.is_manager == False)
            network = AgentNetwork(members=members, formation=_formation, network_tasks=network_tasks)
            return network

        else:
            res = res.json_dict
            formation_keys = [k for k, v in Formation._member_map_.items() if k == res["formation"].upper()]

            if formation_keys:
                _formation = Formation[formation_keys[0]]

            created_agents = [Agent(role=item, goal=item) for item in res["agent_roles"]]
            created_tasks = [Task(description=item) for item in res["task_descriptions"]]

            network_tasks = []
            members = []
            leader = str(res["leader_agent"])

            for i in range(len(created_agents)):
                is_manager = bool(created_agents[i].role.lower() == leader.lower())
                member = Member(agent=created_agents[i], is_manager=is_manager)

                if len(created_tasks) >= i and created_tasks[i]:
                    member.tasks.append(created_tasks[i])

                members.append(member)

            if len(created_agents) < len(created_tasks):
                network_tasks.extend(created_tasks[len(created_agents):len(created_tasks)])

            members.sort(key=lambda x: x.is_manager == False)
            network = AgentNetwork(members=members, formation=_formation,  network_tasks=network_tasks)
            return network


    except Exception as e:
        Logger(verbose=True).log(level="error", message=f"Failed to create a agent network - return None. You can try with solo agent. Error: {str(e)}", color="red")
        return None
