def test_eval():
    import versionhq as vhq
    from pydantic import BaseModel

    class CustomOutput(BaseModel):
        test1: str
        test2: list[str]

    task = vhq.Task(
        description="Research a topic to teach a kid aged 6 about math.",
        pydantic_output=CustomOutput,
        should_evaluate=True, # triggers evaluation
        eval_criteria=["Uniquness", "Fit to audience",],

    )
    res = task.execute()

    assert isinstance(res.evaluation, vhq.Evaluation)
    assert len(res.evaluation.items) == 2
    assert res.evaluation.aggregate_score is not None
    assert res.evaluation.suggestion_summary is not None
