from dotenv import load_dotenv
from sample.demo.agents import agent_a, agent_b1
from sample.demo.tasks import task_1, task_2, team_task
from framework.team.model import Team, TaskHandlingProcess, TeamMember

load_dotenv(override=True)

def demo():
    team = Team(
        members=[
            TeamMember(agent=agent_a, is_manager=True, task=task_1),
            TeamMember(agent=agent_b1, is_manager=False, task=task_2)
        ],
        team_tasks=[team_task,],
        process=TaskHandlingProcess.sequential,
        verbose=True,
        memory=False,
        before_kickoff_callbacks=[], # add any callables
        after_kickoff_callbacks=[],
        prompt_file="sample.demo.Prompts.demo.py"
    )

    team.kickoff()


if __name__ == "__main__":
    demo()
