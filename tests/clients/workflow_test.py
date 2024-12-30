import os
import pytest
from versionhq.agent.model import Agent
from versionhq.llm.model import LLM
from versionhq.clients.workflow.model import Score, ScoreFormat, MessagingWorkflow, MessagingComponent

MODEL_NAME = os.environ.get("LITELLM_MODEL_NAME", "gpt-3.5-turbo")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY")


def test_store_scores():
    """
    Test if the final result will be calcurated using a random subject
    """

    messaging_component = MessagingComponent(message="demo")
    score_raw = 15
    messaging_component.store_scoring_result("demo", score_raw=score_raw)

    assert messaging_component.score is not None
    assert messaging_component.score.result() is not None
