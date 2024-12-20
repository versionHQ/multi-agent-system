import os
from dotenv import load_dotenv
from versionhq.agent.model import Agent
from versionhq.task.model import Task, ResponseField
from versionhq.team.model import Team, TaskHandlingProcess, TeamMember

load_dotenv(override=True)
MODEL_NAME = os.environ.get("LITELLM_MODEL_NAME", "gpt-3.5-turbo")

agent_a = Agent(
    role="Demo Agent A",
    goal="""My amazing goals""",
    backstory="My amazing backstory",
    verbose=True,
    llm=MODEL_NAME,
    max_token=3000
)

agent_b1 =  Agent(
    role="Demo Agent B-1",
    goal="""My amazing goals""",
    backstory="My amazing backstory",
    verbose=True,
    llm=MODEL_NAME,
    max_token=3000
)


task_1 = Task(
    description="Amazing task description",
    expected_output_raw=False,
    expected_output_json=True,
    expected_output_pydantic=True,
    output_field_list=[
        ResponseField(title="field-1", type="int", required=True),
        ResponseField(title="field-2", type="array", required=True) # => { field-1: 3, field-2: [item-1, item2, ]}
    ],
    context=[],
    tools=[],
    callback=None
)

task_2 = Task(
    description="Amazing task description",
    expected_output_raw=False,
    expected_output_json=True,
    expected_output_pydantic=True,
    output_field_list=[
        ResponseField(title="field-1", type="int", required=True),
        ResponseField(title="field-2", type="array", required=True)
    ],
    context=[],
    tools=[],
    callback=None
)

team_task = Task(
    description="Amazing team task description",
    expected_output_raw=False,
    expected_output_json=True,
    expected_output_pydantic=True,
    output_field_list=[
        ResponseField(title="field-1", type="str", required=True),
    ]
)


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
