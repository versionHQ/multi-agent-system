from unittest.mock import patch

from versionhq._utils import UsageMetrics
from versionhq._prompt.model import Prompt


def test_gpt_cua():
    import versionhq as vhq

    params = dict(
        user_prompt="test",
        reasoning_effort="medium",
        browser="firefox",
        tools=vhq.CUAToolSchema(display_width=500, display_height=300, environment="mac", type="computer_use_preview")
    )
    tool = vhq.GPTToolCUA(**params)

    assert tool.user_prompt == params["user_prompt"]
    assert tool.img_url is None
    assert tool.web_url is not None
    assert tool.browser == params["browser"]
    assert tool.reasoning_effort == "medium"
    assert isinstance(tool.tools, list)
    assert tool.tools[0].display_width == 500
    assert tool.tools[0].type == "computer_use_preview"

    with patch.object(vhq.GPTToolCUA, "_structure_schema", return_value=None) as mock_schema:
        tool.run()
        mock_schema.assert_called()


def test_gpt_web_search():
    import versionhq as vhq

    params = dict(input="Search today's top news.", country="US", city="test", search_content_size = "high")
    tool = vhq.GPTToolWebSearch(**params)
    assert tool.model == "gpt-4o"
    assert tool.input == params["input"]
    assert tool._user_location == None
    assert isinstance(tool._usage, UsageMetrics)
    assert tool.search_content_size == "high"
    assert tool.schema is not None

    raw, annotations, usage = tool.run()
    assert raw is not None if usage.total_errors == 0 else raw == ""

    if raw:
        assert isinstance(annotations, list)
        assert isinstance(usage, UsageMetrics)
        assert usage.total_tokens > 0
        assert usage.latency > 0


def test_gpt_file_search():
    import versionhq as vhq

    params = dict(
        input="Search today's top news.",
        vector_store_ids="vs_dummy_id",
        max_num_results=5,
    )
    tool = vhq.GPTToolFileSearch(**params)
    assert tool.model == "gpt-4o"
    assert tool.input == params["input"]
    assert isinstance(tool._usage, UsageMetrics)
    assert tool.vector_store_ids == [params["vector_store_ids"]]
    assert tool.max_num_results == params["max_num_results"]
    assert tool.schema is not None

    with patch.object(vhq.GPTToolFileSearch, "run", return_value=("test", [], UsageMetrics())) as mock_run:
        tool.run()
        mock_run.assert_called_once()


def test_with_agent():
    import versionhq as vhq

    tool_1 = vhq.GPTToolWebSearch(input="Search today's top news.", search_content_size="high")
    tool_2 = vhq.GPTToolFileSearch(input="Search today's top news.", vector_store_ids="vs_dummy_id", max_num_results=5)

    agent = vhq.Agent(role="GPT Tool Handling", tools=[tool_1])
    with patch.object(vhq.Agent, "_handle_gpt_tools", return_value=(vhq.TaskOutput(raw=""))) as mock_run:
        agent.start(tool_res_as_final=True)
        mock_run.assert_called_once()

    with patch.object(Prompt, "format_core", return_value=("test", "test", ["test"])) as mock_prompt:
        agent.start()
        mock_prompt.assert_called_once()


    agent = vhq.Agent(role="GPT Tool Handling", tools=[tool_2])
    with patch.object(vhq.Agent, "_handle_gpt_tools", return_value=(vhq.TaskOutput(raw=""))) as mock_run:
        agent.start(tool_res_as_final=True)
        mock_run.assert_called_once()

    with patch.object(Prompt, "format_core", return_value=("test", "test", ["test"])) as mock_prompt:
        agent.start()
        mock_prompt.assert_called_once()


    agent = vhq.Agent(role="GPT Tool Handling", tools=[tool_1, tool_2])
    with patch.object(vhq.Agent, "_handle_gpt_tools", return_value=("test", "test", vhq.TaskOutput(raw=""), UsageMetrics())) as mock_run:
        agent.start(tool_res_as_final=True)
        mock_run.assert_called_once()

    with patch.object(Prompt, "format_core", return_value=("test", "test", ["test"])) as mock_prompt:
        agent.start()
        mock_prompt.assert_called_once()


def test_with_task():
    import versionhq as vhq

    tool_1 = vhq.GPTToolWebSearch(input="Search today's top news.", search_content_size = "high")
    tool_2 = vhq.GPTToolFileSearch(input="Search today's top news.", vector_store_ids="vs_dummy_id", max_num_results=5)

    task = vhq.Task(description="Test gpt tools", tools=[tool_1], tool_res_as_final=True)
    with patch.object(vhq.Agent, "_handle_gpt_tools", return_value=(vhq.TaskOutput(raw="", usage=UsageMetrics()))) as mock_run:
        task.execute()
        mock_run.assert_called_once()

    with patch.object(Prompt, "format_core", return_value=("test", "test", ["test"])) as mock_prompt:
        task.tool_res_as_final = False
        task.execute()
        mock_prompt.assert_called_once()


    task = vhq.Task(description="Test gpt tools", tools=[tool_2], tool_res_as_final=True)
    with patch.object(vhq.Agent, "_handle_gpt_tools", return_value=(vhq.TaskOutput(raw="", usage=UsageMetrics()))) as mock_run:
        task.execute()
        mock_run.assert_called_once()

    with patch.object(Prompt, "format_core", return_value=("test", "test", ["test"])) as mock_prompt:
        task.tool_res_as_final = False
        task.execute()
        mock_prompt.assert_called_once()


    task = vhq.Task(description="Test gpt tools", tools=[tool_1, tool_2], tool_res_as_final=True)
    with patch.object(vhq.Agent, "_handle_gpt_tools", return_value=(vhq.TaskOutput(raw="", usage=UsageMetrics()))) as mock_run:
        task.execute()
        mock_run.assert_called_once()

    with patch.object(Prompt, "format_core", return_value=("test", "test", ["test"])) as mock_prompt:
        task.tool_res_as_final = False
        task.execute()
        mock_prompt.assert_called_once()
