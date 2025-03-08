from typing import List, Type
from enum import Enum

from pydantic import BaseModel, create_model, Field

from versionhq.task.model import Task
from versionhq.agent.model import Agent
from versionhq.agent_network.model import AgentNetwork, Member, Formation
from versionhq.agent.inhouse_agents  import vhq_formation_planner
from versionhq._utils import Logger

import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()


def form_agent_network(
        task: str,
        expected_outcome: str | Type[BaseModel],
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
                    pass

                case str():
                    matched = [item for item in Formation.s_ if item == formation.upper()]
                    if matched:
                        formation = getattr(Formation, matched[0])
                    else:
                        # Formation._generate_next_value_(name=f"CUSTOM_{formation.upper()}", start=100, count=6, last_values=Formation.HYBRID.name)
                        Logger(verbose=True).log(level="warning", message=f"The formation {formation} is invalid. We'll recreate a valid formation.", color="yellow")
                        formation = None

                case int():
                    formation = Formation(int(formation))

                case float():
                    formation = Formation(round(formation))

                case _:
                    Logger(verbose=True).log(level="warning", message=f"The formation {formation} is invalid. We'll recreate a valid formation.", color="yellow")
                    formation = None

        except Exception as e:
            Logger(verbose=True).log(level="warning", message=f"The formation {formation} is invalid: {str(e)}. We'll recreate a formation.", color="yellow")
            formation = None

    # try:
    prompt_formation = formation.name if formation and isinstance(formation, Formation) else f"Select the best formation to effectively execute the tasks from the given Enum sets: {str(Formation.__dict__)}."

    prompt_expected_outcome = expected_outcome if isinstance(expected_outcome, str) else str(expected_outcome.model_dump()) if type(expected_outcome) == BaseModel else ""

    class Outcome(BaseModel):
        formation: str
        agent_roles: list[str]
        task_descriptions: list[str]
        task_outcomes: list[list[str]]
        leader_agent: str

    vhq_task = Task(
        description=f"Design a team of specialized agents to fully automate the following task and achieve the expected outcome. For each agent, define its role, task description, and expected outputs via the task with items in a list. Then specify the formation if the formation is not given. If you think SUPERVISING or HYBRID is the best formation, include a leader_agent role, else leave the leader_agent role blank.\nTask: {str(task)}\nExpected outcome: {prompt_expected_outcome}\nFormation: {prompt_formation}",
        pydantic_output=Outcome
    )

    if agents:
        vhq_task.description += "You MUST add the following agents' roles in the network formation: " + ", ".join([agent.role for agent in agents if isinstance(agent, Agent)])

    res = vhq_task.execute(agent=vhq_formation_planner, context=context)

    formation_keys = []
    if hasattr(res.pydantic, "formation"):
        formation_keys = [k for k in Formation._member_map_.keys() if k == res.pydantic.formation.upper()]
    elif "formation" in res.json_dict:
        formation_keys = [k for k in Formation._member_map_.keys() if k == res.json_dict["formation"].upper()]

    _formation = Formation[formation_keys[0]] if formation_keys else Formation.SUPERVISING

    network_tasks = []
    members = []
    leader = str(res.pydantic.leader_agent) if res.pydantic else str(res.json_dict["leader_agent"])

    agent_roles = res.pydantic.agent_roles if res.pydantic else res.json_dict["agent_roles"]
    created_agents = [Agent(role=str(item), goal=str(item)) for item in agent_roles]

    if agents:
        for i, agent in enumerate(created_agents):
            matches = [item for item in agents if item.role == agent.role]
            if matches:
                created_agents[i] = matches[0]
            else:
                pass

    created_tasks = []

    if res.pydantic:
        for i, item in enumerate(res.pydantic.task_outcomes):
            if len(res.pydantic.task_descriptions) > i and res.pydantic.task_descriptions[i]:
                fields = {}
                for ob in item:
                    try:
                        field_name = str(ob).lower().split(":")[0].replace(" ", "_")[0: 16]
                        fields[field_name] = (str, Field(default=None))
                    except:
                        pass
                output = create_model("Output", **fields) if fields else None
                _task = Task(description=res.pydantic.task_descriptions[i], pydantic_output=output)
                created_tasks.append(_task)

    elif res.json_dict:
        for i, item in enumerate(res["task_outcomes"]):
            if len(res["task_descriptions"]) > i and res["task_descriptions"][i]:
                fields = {}
                for ob in item:
                    try:
                        field_name = str(ob).lower().split(":")[0].replace(" ", "_")[0: 16]
                        fields[field_name] = (str, Field(default=None))
                    except:
                        pass
                output = create_model("Output", **fields) if fields else None
                _task = Task(description=res["task_descriptions"][i], pydantic_output=output)
                created_tasks.append(_task)


    if len(created_tasks) <= len(created_agents):
        for i in range(len(created_tasks)):
            is_manager = bool(created_agents[i].role.lower() == leader.lower())
            member = Member(agent=created_agents[i], is_manager=is_manager, tasks=[created_tasks[i]])
            members.append(member)

        for i in range(len(created_tasks), len(created_agents)):
            try:
                is_manager = bool(created_agents[i].role.lower() == leader.lower())
                member_w_o_task = Member(agent=created_agents[i], is_manager=is_manager)
                members.append(member_w_o_task)
            except:
                pass

    elif len(created_tasks) > len(created_agents):
        for i in range(len(created_agents)):
            is_manager = bool(created_agents[i].role.lower() == leader.lower())
            member = Member(agent=created_agents[i], is_manager=is_manager, tasks=[created_tasks[i]])
            members.append(member)

        network_tasks.extend(created_tasks[len(created_agents):len(created_tasks)])


    if _formation == Formation.SUPERVISING and not [member for member in members if member.is_manager]:
        manager = Member(agent=Agent(role=leader, goal=leader), is_manager=True)
        members.append(manager)

    members.sort(key=lambda x: x.is_manager == False)
    network = AgentNetwork(members=members, formation=_formation, network_tasks=network_tasks)

    Logger().log(level="info", message=f"Successfully created a agent network: {str(network.id)} with {len(network.members)} agents.", color="blue")

    return network


    # except Exception as e:
    #     Logger().log(level="error", message=f"Failed to create a agent network - return None. You can try with solo agent. Error: {str(e)}", color="red")
    #     return None
