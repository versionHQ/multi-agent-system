from unittest.mock import patch


def test_eval():
    import versionhq as vhq
    from pydantic import BaseModel

    class CustomOutput(BaseModel):
        topics: list[str]
        reasons: list[str]

    task = vhq.Task(
        description="Research a topic to teach a kid aged 6 about math.",
        pydantic_output=CustomOutput,
        should_evaluate=True,
        eval_criteria=["Uniquness", "Fit to audience",],
        # fsls=[""]
    )


    with patch.object(vhq.Evaluation, "_draft_fsl_prompt", return_value="test") as draft_fsl_prompt:
        res = task.execute()
        assert isinstance(res.evaluation, vhq.Evaluation)
        assert [isinstance(item, vhq.EvaluationItem) and item.criteria in task.eval_criteria for item in res.evaluation.items]
        assert res.latency and res._tokens
        assert res.evaluation.aggregate_score is not None
        assert res.evaluation.suggestion_summary

        draft_fsl_prompt.assert_called_once_with(task_description=task.description)


def test_eval_with_fsls():
    import versionhq as vhq
    from pydantic import BaseModel

    class CustomOutput(BaseModel):
        topics: list[str]
        reasons: list[str]

    task = vhq.Task(
        description="Research a topic to teach a kid aged 6 about math.",
        pydantic_output=CustomOutput,
        should_evaluate=True,
        eval_criteria=["Uniquness", "Fit to audience",],
        fsls=["Start by explaining that math is all around us and helps us solve problems in our daily lives. Today, we're going to learn about adding and subtracting using objects.", "Focus on complex concepts like seconds and milliseconds."]
    )

    res = task.execute()
    assert isinstance(res.evaluation, vhq.Evaluation)
    assert [isinstance(item, vhq.EvaluationItem) and item.criteria in task.eval_criteria for item in res.evaluation.items]
    assert res.latency and res._tokens
    assert res.evaluation.aggregate_score is not None
    assert res.evaluation.suggestion_summary is not None
