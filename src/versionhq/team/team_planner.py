import os
from typing import Any, List, Optional, Dict
from pydantic import BaseModel, Field




class TeamPlanner:
    """
    A class to handle agent formations based on the given task description.
    1) complexity
    2) agent built (or use given agents)
    3) knowledge, memory sharing
    4) form a team
    """

    from versionhq.task.model import Task, ResponseField, TaskOutput
    from versionhq.agent.model import Agent


    def __init__(self, tasks: List[Task], planner_llm: Optional[Any] = None):
        self.tasks = tasks
        self.planner_llm = planner_llm if planner_llm else os.environ.get("DEFAULT_MODEL_NAME")


    def _handle_assign_agents(self, unassigned_tasks: List[Task]) -> List[Any]:
        """
        Build an agent and assign it a task, then return a list of Member connecting the agent created and the task given.
        """

        from versionhq.agent.model import Agent
        from versionhq.task.model import Task, ResponseField
        from versionhq.team.model import Member

        new_member_list: List[Member] = []
        agent_creator = Agent(
            role="agent_creator",
            goal="build an ai agent that can competitively handle the task given",
            llm=self.planner_llm,
        )

        for unassgined_task in unassigned_tasks:
            task = Task(
                description=f"""
                    Based on the following task summary, draft a AI agent's role and goal in concise manner.
                    Task summary: {unassgined_task.summary}
                """,
                response_fields=[
                    ResponseField(title="goal", data_type=str, required=True),
                    ResponseField(title="role", data_type=str, required=True),
                ],
            )
            res = task.execute_sync(agent=agent_creator)
            agent = Agent(
                role=res.json_dict["role"] if "role" in res.json_dict else res.raw,
                goal=res.json_dict["goal"] if "goal" in res.json_dict else task.description
            )
            if agent.id:
                team_member = Member(agent=agent, tasks=[unassgined_task], is_manager=False)
                new_member_list.append(team_member)

        return new_member_list


    def _handle_task_planning(self, context: Optional[str] = None, tools: Optional[str] = None) -> TaskOutput:
        """
        Handles the team planning by creating detailed step-by-step plans for each task.
        """

        from versionhq.agent.model import Agent
        from versionhq.task.model import Task

        team_planner = Agent(
            role="team planner",
            goal="Plan extremely detailed, step-by-step plan based on the tasks and tools available to each agent so that they can perform the tasks in an exemplary manner and assign a task to each agent.",
            llm=self.planner_llm,
        )

        task_summary_list = [task.summary for task in self.tasks]

        class TeamPlanIdea(BaseModel):
            plan: str | Dict[str, Any] = Field(default=None, description="a decriptive plan to be executed by the team")


        task = Task(
            description=f"""
                Based on the following task summaries, create the most descriptive plan that the team can execute most efficiently. Take all the task summaries - task's description and tools available - into consideration. Your answer only contains a dictionary.

                Task summaries: {" ".join(task_summary_list)}
            """,
            pydantic_output=TeamPlanIdea
        )
        output = task.execute_sync(agent=team_planner, context=context, tools=tools)
        return output
