"""Test code snippets on docs/core/task.md"""


def test_docs_core_task_a():
    import versionhq as vhq

    task = vhq.Task(description="MY AMAZING TASK")

    import uuid
    assert uuid.UUID(str(task.id), version=4)


def test_docs_core_task_b():
    import versionhq as vhq

    task = vhq.Task(description="MY AMAZING TASK")
    res = task.execute()

    assert isinstance(res, vhq.TaskOutput)
    assert res.raw and res.json
    assert task.processed_agents is not None



def test_docs_core_task_c():
    from pydantic import BaseModel
    from typing import Any

    class Demo(BaseModel):
        """
        A demo pydantic class to validate the outcome with various nested data types.
        """
        demo_1: int
        demo_2: float
        demo_3: str
        demo_4: bool
        demo_5: list[str]
        demo_6: dict[str, Any]
        demo_nest_1: list[dict[str, Any]] # 1 layer of nested child is ok.
        demo_nest_2: list[list[str]]
        demo_nest_3: dict[str, list[str]]
        demo_nest_4: dict[str, dict[str, Any]]
        # error_1: list[list[dict[str, list[str]]]] # <- Error due to 2+ layers of nested children.
        # error_2: InstanceOf[AnotherPydanticClass] # <- Error due to non-typing annotation.

    # 2. Define a task
    import versionhq as vhq
    task = vhq.Task(
        description="generate random output that strictly follows the given format",
        pydantic_output=Demo,
    )

    # 3. Execute
    res = task.execute()

    assert isinstance(res, vhq.TaskOutput) and res.task_id is task.id
    assert isinstance(res.raw, str) and isinstance(res.json_dict, dict)
    assert [
        getattr(res.pydantic, k) and v.annotation == Demo.model_fields[k].annotation
        for k, v in res.pydantic.model_fields.items()
    ]



def test_docs_core_task_d():
    import versionhq as vhq

    # 1. Define a list of ResponseField objects
    demo_response_fields = [
        vhq.ResponseField(title="demo_1", data_type=int),
        vhq.ResponseField(title="demo_2", data_type=float),
        vhq.ResponseField(title="demo_3", data_type=str),
        vhq.ResponseField(title="demo_4", data_type=bool),
        vhq.ResponseField(title="demo_5", data_type=list, items=str),
        vhq.ResponseField(title="demo_6", data_type=dict, properties=[vhq.ResponseField(title="demo-item", data_type=str)]),
        vhq.ResponseField(title="demo_nest_1", data_type=list, items=dict, properties=([
            vhq.ResponseField(title="nest1", data_type=dict, properties=[vhq.ResponseField(title="nest11", data_type=str)])
        ])),
        vhq.ResponseField(title="demo_nest_2", data_type=list, items=list),
        vhq.ResponseField(title="demo_nest_3", data_type=dict, properties=[
            vhq.ResponseField(title="nest1", data_type=list, items=str)
        ]),
        vhq.ResponseField(title="demo_nest_4", data_type=dict, properties=[
            vhq.ResponseField(title="nest1", data_type=dict, properties=[vhq.ResponseField(title="nest12", data_type=str)])
        ])
    ]

    task = vhq.Task(
        description="Output random values strictly following the data type defined in the given response format.",
        response_fields=demo_response_fields
    )

    res = task.execute()

    assert isinstance(res, vhq.TaskOutput) and res.task_id is task.id
    assert res.raw and res.json and res.pydantic is None
    assert [v and type(v) == task.response_fields[i].data_type for i, (k, v) in enumerate(res.json_dict.items())]



def test_docs_core_task_d():
    import versionhq as vhq
    from pydantic import BaseModel
    from typing import Any

    # 1. Define and execute a sub task with Pydantic output.
    class Sub(BaseModel):
        sub1: list[dict[str, Any]]
        sub2: dict[str, Any]

    sub_task = vhq.Task(
        description="generate random values that strictly follows the given format.",
        pydantic_output=Sub
    )
    sub_res = sub_task.execute()

    # 2. Define a main task, callback function to format the final response.
    class Main(BaseModel):
        main1: list[Any] # <= assume expecting to store Sub object.
        main2: dict[str, Any]

    def format_response(sub, main1, main2) -> Main:
        main1.append(sub)
        main = Main(main1=main1, main2=main2)
        return main

    # 3. Execute
    main_task = vhq.Task(
        description="generate random values that strictly follows the given format.",
        pydantic_output=Main,
        callback=format_response,
        callback_kwargs=dict(sub=Sub(sub1=sub_res.pydantic.sub1, sub2=sub_res.pydantic.sub2)),
    )
    res = main_task.execute(context=sub_res.raw) # [Optional] Adding sub_task's response as context.

    assert [item for item in res.callback_output.main1 if isinstance(item, Sub)]
