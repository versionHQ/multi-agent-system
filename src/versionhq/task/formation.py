from typing import List
from enum import Enum

from pydantic import BaseModel

from versionhq.task.model import Task
from versionhq.agent.model import Agent
from versionhq.team.model import Team, TeamMember, Formation
from versionhq.agent.inhouse_agents  import vhq_formation_planner
from versionhq._utils import Logger


def form_agent_network(
        task_overview: str,
        expected_outcome: str,
        agents: List[Agent] = None,
        context: str = None,
        formation: Formation = None
    ) -> Team | None:
    """
    Make a formation of agents from the given task description, agents (optional), context (optional), and expected outcome (optional).
    """

    if not task_overview:
        Logger(verbose=True).log(level="error", message="Missing task description.", color="red")
        return None

    if not expected_outcome:
        Logger(verbose=True).log(level="error", message="Missing expected outcome.", color="red")
        return None


    try:
        class Outcome(BaseModel):
            formation: Enum
            agent_roles: list[str]
            task_descriptions: list[str]
            leader_agent: str

        vhq_task = Task(
            description=f"""
    Create a team of specialized agents designed to automate the following task and deliver the expected outcome. Consider the necessary roles for each agent with a clear task description. If you think we neeed a leader to handle the automation, return a leader_agent role as well, but if not, leave the a leader_agent role blank.
    Task: {str(task_overview)}
    Expected outcome: {str(expected_outcome)}
            """,
            pydantic_output=Outcome
        )

        if formation:
            vhq_task.description += f"Select 1 formation you think the best from the given Enum sets: {str(Formation.__dict__)}"

        if agents:
            vhq_task.description += "Consider adding following agents in the formation: " + ", ".join([agent.role for agent in agents if isinstance(agent, Agent)])

        res = vhq_task.execute_sync(agent=vhq_formation_planner, context=context)
        formation_ = Formation.SUPERVISING

        if res.pydantic:
            formation_keys = [k for k, v in Formation._member_map_.items() if k == res.pydantic.formation.upper()]

            if formation_keys:
                formation_ = Formation[formation_keys[0]]

            created_agents = [Agent(role=item, goal=item) for item in res.pydantic.agent_roles]
            created_tasks = [Task(description=item) for item in res.pydantic.task_descriptions]
            team_tasks = []
            members = []
            leader = str(res.pydantic.leader_agent)

            for i in range(len(created_agents)):
                is_manager = bool(created_agents[i].role.lower() in leader.lower())
                member = TeamMember(agent=created_agents[i], is_manager=is_manager)

                if len(created_tasks) >= i:
                    member.task = created_tasks[i]
                    members.append(member)

            if len(created_agents) < len(created_tasks):
                team_tasks.extend(created_tasks[len(created_agents) - 1:len(created_tasks)])

            members.sort(key=lambda x: x.is_manager == False)
            team = Team(members=members, formation=formation_)
            return team

        else:
            formation_keys = [k for k, v in Formation._member_map_.items() if k == res.json_dict["formation"].upper()]

            if formation_keys:
                formation_ = Formation[formation_keys[0]]

            created_agents = [Agent(role=item, goal=item) for item in res.json_dict["agent_roles"]]
            created_tasks = [Task(description=item) for item in res.json_dict["task_descriptions"]]
            team_tasks = []
            members = []
            leader = str(res.json_dict["leader_agent"])

            for i in range(len(created_agents)):
                is_manager = bool(created_agents[i].role.lower() in leader.lower())
                member = TeamMember(agent=created_agents[i], is_manager=is_manager)

                if len(created_tasks) >= i:
                    member.task = created_tasks[i]
                    members.append(member)

            if len(created_agents) < len(created_tasks):
                team_tasks.extend(created_tasks[len(created_agents) - 1:len(created_tasks)])

            members.sort(key=lambda x: x.is_manager == True)
            team = Team(members=members, formation=formation_)
            return team

    except Exception as e:
        Logger(verbose=True).log(level="error", message=f"Failed to create an agent network - return None. You can try with solo agent. Error: {str(e)}", color="red")
        return None



if __name__ == "__main__":
    res = form_agent_network(
        task_overview="Launch an outbound campaign to attract young audience.",
        expected_outcome="Best media mix of the campaign.",
        context="We are selling sports wear.",
    )
