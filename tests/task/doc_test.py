"""Test code snippets on docs/core/task.md"""

from versionhq._utils import UsageMetrics
from versionhq._prompt.auto_feedback import PromptFeedbackGraph


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
    import versionhq as vhq
    from tests import Demo

    task = vhq.Task(
        description="generate random output that strictly follows the given format",
        response_schema=Demo,
    )
    res = task.execute()

    assert isinstance(res, vhq.TaskOutput) and res.task_id is task.id
    assert isinstance(res.raw, str) and isinstance(res.json_dict, dict)
    assert [
        getattr(res.pydantic, k) is not None and v.annotation == Demo.model_fields[k].annotation
        for k, v in res.pydantic.model_fields.items()
    ]


def test_docs_core_task_d():
    import versionhq as vhq
    from tests import demo_response_fields

    task = vhq.Task(
        description="Output random values strictly following the data type defined in the given response format.",
        response_schema=demo_response_fields
    )
    res = task.execute()

    assert isinstance(res, vhq.TaskOutput) and res.task_id is task.id
    assert res.raw is not None and res.json_dict is not None
    assert res.pydantic is None
    assert [v and type(v) == task.response_schema[i].data_type for i, (k, v) in enumerate(res.json_dict.items())]


def test_docs_core_task_e():
    import versionhq as vhq
    from pydantic import BaseModel
    from typing import Any

    # 1. Define and execute a sub task with Pydantic output.
    class Sub(BaseModel):
        sub1: str
        sub2: dict[str, Any]

    sub_task = vhq.Task(description="generates a random value that strictly follows the given format.", response_schema=Sub)
    sub_res = sub_task.execute()

    class Main(BaseModel):
        main1: list[Any]
        main2: str

    def format_response(sub, main1, main2) -> Main:
        if main1:
            main1.append(sub)
        main = Main(main1=main1, main2=str(main2))
        return main

    main_task = vhq.Task(
        description="generates a random value that strictly follows the givne format.",
        response_schema=Main,
        callback=format_response,
        callback_kwargs=dict(sub=sub_res.json_dict),
    )
    res = main_task.execute(context=sub_res.raw) # [Optional] Adding sub_task's response as context.
    assert res.callback_output.main1 is not None
    assert res.callback_output.main2 is not None


def test_docs_core_task_g():
    import versionhq as vhq
    task = vhq.Task(
        description="return the output following the given prompt.",
        allow_delegation=True
    )
    task.execute()

    assert task.output is not None
    assert task.processed_agents is not None
    assert task._delegations ==1


def test_docs_core_task_h():
    import versionhq as vhq

    task = vhq.Task(description="Return a word: 'test'", type=vhq.TaskExecutionType.ASYNC)

    from unittest.mock import patch
    with patch.object(vhq.Agent, "execute_task", return_value=("user prompt", "dev prompt", "test")) as execute:
        res = task.execute()
        assert res.raw == "test"
        execute.assert_called_once_with(task=task, context=None, task_tools=list())


def test_docs_core_task_i():
    import versionhq as vhq

    def random_func(message: str) -> str:
        return message + "_demo"

    tool = vhq.Tool(name="tool", func=random_func)
    tool_set = vhq.ToolSet(tool=tool, kwargs=dict(message="empty func"))
    task = vhq.Task(
        description="execute the given tools",
        tools=[tool_set,],
        tool_res_as_final=True
    )

    res = task.execute()
    assert res.tool_output == "empty func_demo"


def test_docs_core_task_j():
    import versionhq as vhq

    simple_tool = vhq.Tool(name="simple tool", func=lambda x: "simple func")
    agent = vhq.Agent(role="demo", goal="execute tools", tools=[simple_tool,])
    task = vhq.Task(
        description="execute tools",
        can_use_agent_tools=True,
        tool_res_as_final=True
    )
    res = task.execute(agent=agent)
    assert res.tool_output == "simple func"


def test_docs_core_task_k():
    import versionhq as vhq

    def callback_func(condition: str, test1: str):
        return f"Result: {test1}, condition added: {condition}"

    task = vhq.Task(
        description="return the output following the given prompt.",
        callback=callback_func,
        callback_kwargs=dict(condition="demo for pytest")
    )
    res = task.execute()

    assert res and isinstance(res, vhq.TaskOutput)
    assert res.task_id is task.id
    assert "demo for pytest" in res.callback_output


def test_docs_core_task_l():
    import versionhq as vhq
    from pathlib import Path

    current_dir = Path(__file__).parent.parent
    file_path = current_dir / "_sample/screenshot.png"
    audio_path = current_dir / "_sample/sample.mp3"

    task = vhq.Task(description="Summarize the given content", image=str(file_path), audio=str(audio_path))
    res = task.execute(agent=vhq.Agent(llm="gemini-2.0", role="Content Interpretator"))

    assert res.final is not None


def test_docs_core_task_m():
    import versionhq as vhq

    task = vhq.Task(description="Create a short story.", should_test_run=True, human=False)
    assert isinstance(task._usage, UsageMetrics)

    task.execute()
    assert isinstance(task._pfg, PromptFeedbackGraph)
    assert task._pfg.should_reform == False
    assert task._pfg.reform_trigger_event == None
    assert [k for k in task._pfg.user_prompts.keys()] and [k for k in task._pfg.dev_prompts.keys()]

    assert task._pfg._usage.total_tokens == task._usage.total_tokens
    assert task._usage.total_errors == task._pfg._usage.total_errors
    assert task._usage.latency == task._pfg._usage.latency
