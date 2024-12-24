import os
from dotenv import load_dotenv
from typing import Any, List, Optional
from pydantic import BaseModel, Field

from versionhq.agent.model import Agent
from versionhq.task.model import Task, ResponseField

load_dotenv(override=True)


class TeamPlanner:
    """
    (Optional) Plan how the team should handle multiple tasks using LLM.
    """

    def __init__(self, tasks: List[Task], planner_llm: Optional[Any] = None):
        self.tasks = tasks
        self.planner_llm = (
            planner_llm if planner_llm != None else os.environ.get("LITELLM_MODEL_NAME")
        )

    def _handle_task_planning(self) -> BaseModel:
        """
        Handles the team planning by creating detailed step-by-step plans for each task.
        """

        planning_agent = Agent(
            role="Task Execution Planner",
            goal="Your goal is to create an extremely detailed, step-by-step plan based on the tasks and tools available to each agent so that they can perform the tasks in an exemplary manner",
            backstory="You have a strong ability to design efficient organizational structures and task processes, minimizing unnecessary steps.",
            llm=self.planner_llm,
        )

        task_summary_list = [task.summary for task in self.tasks]
        task_to_handle = Task(
            description=f"""
                Based on the following task summaries, create the most descriptive plan that the team can execute most efficiently. Take all the task summaries - task's description and tools available - into consideration. Your answer only contains a dictionary.

                Task summaries: {" ".join(task_summary_list)}
             """,
            expected_output_json=False,
            expected_output_pydantic=True,
            output_field_list=[
                ResponseField(title=f"{task.id}", type=str, required=True)
                for task in self.tasks
            ],
        )
        task_output = task_to_handle.execute_sync(agent=planning_agent)

        if isinstance(task_output.pydantic, BaseModel):
            return task_output.pydantic

        else:
            return None
