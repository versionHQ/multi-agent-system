# import os
# from pydantic import BaseModel

# from versionhq.agent.model import Agent
# from versionhq.task.model import Task, ResponseField, TaskOutput
# from versionhq.team.model import Team, TeamMember, TaskHandlingProcess, TeamOutput
# from versionhq._utils.usage_metrics import UsageMetrics

# MODEL_NAME = os.environ.get("LITELLM_MODEL_NAME", "gpt-3.5-turbo")

# def test_kickoff():
#     agent_a = Agent(
#         role="Demo Agent A",
#         goal="""My amazing goals""",
#         backstory="My amazing backstory",
#         verbose=True,
#         llm=MODEL_NAME,
#         max_tokens=3000,
#     )

#     agent_b1 = Agent(
#         role="Demo Agent B-1",
#         goal="""My amazing goals""",
#         backstory="My amazing backstory",
#         verbose=True,
#         llm=MODEL_NAME,
#         max_tokens=3000,
#     )

#     task_1 = Task(
#         description="Amazing task description",
#         expected_output_json=True,
#         expected_output_pydantic=True,
#         output_field_list=[
#             ResponseField(title="field-1", type=int, required=True),
#             ResponseField(
#                 title="field-2", type=list, required=True
#             ),  # => { field-1: 3, field-2: [item-1, item2, ]}
#         ],
#         context=[],
#         tools=[],
#         callback=None,
#     )

#     task_2 = Task(
#         description="Amazing task description",
#         expected_output_json=True,
#         expected_output_pydantic=True,
#         output_field_list=[
#             ResponseField(title="field-1", type=int, required=True),
#             ResponseField(title="field-2", type=list, required=True),
#         ],
#         context=[],
#         tools=[],
#         callback=None,
#     )

#     team_task = Task(
#         description="Amazing team task description",
#         expected_output_json=True,
#         expected_output_pydantic=True,
#         output_field_list=[
#             ResponseField(title="field-1", type=str, required=True),
#         ],
#     )

#     team = Team(
#         members=[
#             TeamMember(agent=agent_a, is_manager=True, task=task_1),
#             TeamMember(agent=agent_b1, is_manager=False, task=task_2),
#         ],
#         team_tasks=[team_task,],
#         process=TaskHandlingProcess.sequential,
#         verbose=True,
#         memory=False,
#         before_kickoff_callbacks=[],  # add any callables
#         after_kickoff_callbacks=[],
#         prompt_file="sample.demo.Prompts.demo.py",
#     )

#     res = team.kickoff()
#     res_all = res.return_all_task_outputs()

#     assert isinstance(res, TeamOutput)
#     assert res.team_id == team.id
#     assert res.raw is not None
#     assert isinstance(res.raw, str)
#     assert res.json_dict is not None
#     assert isinstance(res.json_dict, dict)
#     assert isinstance(res.pydantic, BaseModel)

#     assert isinstance(res_all, list)
#     assert len(res_all) == 2
#     for item in res_all:
#         assert isinstance(item, TaskOutput)
#         assert "field-1" in item
#         assert "field-2" in item

#     assert isinstance(res.token_usage, UsageMetrics)
#     assert res.token_usage.total_tokens > 0
