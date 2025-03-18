def test_doc_core_taskoutput_a():
    import versionhq as vhq
    from pydantic import BaseModel

    class CustomOutput(BaseModel):
        test1: str
        test2: list[str]

    def dummy_tool():
        return "dummy"

    def summarize_response(message: str, **kwargs) -> str | None:
        test1 = kwargs["test1"] if kwargs and "test1" in kwargs else None
        test2 = kwargs["test2"] if kwargs and "test2" in kwargs else None
        if test1 and test2:
            return f"""{message}: {test1}, {", ".join(str(test2))}"""
        else:
            return None

    task = vhq.Task(
        description="Research a topic to teach a kid aged 6 about math.",
        response_schema=CustomOutput,
        tools=[dummy_tool],
        callback=summarize_response,
        callback_kwargs=dict(message="Hi! Here is the result: "),
        should_evaluate=True, # triggers evaluation
        eval_criteria=["Uniquness", "Fit to audience",],

    )
    res = task.execute()

    assert res.task_id == task.id
    assert res.raw
    assert res.json_dict
    assert res.pydantic.test1 and res.pydantic.test2
    if res.callback_output:
        assert "Hi! Here is the result: " in res.callback_output
    assert res.tool_output is None
    assert res.evaluation and isinstance(res.evaluation, vhq.Evaluation)
