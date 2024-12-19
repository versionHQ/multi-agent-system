from framework.task.model import Task, ResponseField

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
