def test_doc_core_taskoutput_a():
    import versionhq as vhq
    from pydantic import BaseModel

    class CustomOutput(BaseModel):
        test1: str
        test2: list[str]

    def mock_tool():
        return "dummy"

    def mock_func(message: str, test1: str, test2: list[str]) -> str:
        return f"""{message}: {test1}, {", ".join(test2)}"""

    task = vhq.Task(
        description="Research a topic to teach a kid aged 6 about math.",
        response_schema=CustomOutput,
        tools=[mock_tool],
        callback=mock_func,
        callback_kwargs=dict(message="Hi! Here is the result: "),
        should_evaluate=True,
        eval_criteria=["Uniquness", "Fit to audience",],
    )
    res = task.execute()

    assert res.task_id == task.id
    assert res.raw
    assert res.json_dict
    assert res.pydantic.test1 and res.pydantic.test2
    assert "Hi! Here is the result: " in res.callback_output
    assert res.pydantic.test1 in res.callback_output and ", ".join(res.pydantic.test2) in res.callback_output
    assert res.tool_output is None
    assert isinstance(res.evaluation, vhq.Evaluation)
    assert [isinstance(item, vhq.EvaluationItem) and item.criteria in task.eval_criteria for item in res.evaluation.items]
    assert res.evaluation.aggregate_score is not None and res.evaluation.suggestion_summary
    assert res.final == res.callback_output
