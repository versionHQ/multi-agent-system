import os
import pytest
import datetime

from versionhq.agent.model import Agent
from versionhq.llm.model import LLM
from versionhq.clients.workflow.model import Score, ScoreFormat, MessagingWorkflow, MessagingComponent
from versionhq.clients.product.model import Product, ProductProvider


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


def test_score_result():
    messaging_component = MessagingComponent(message="demo")
    score_raw = 15
    messaging_component.store_scoring_result("demo", score_raw=score_raw)

    result = messaging_component.score.result()

    assert result is not None
    assert result != 0


def test_setup_messaging_workflow_with_anonymous_provider():
    product = Product(
        description="demo p",
        audience="demo audience",
        usp="demo usp",
        landing_url="www.com",
        cohort_timeframe=30,
    )
    comp = MessagingComponent(message="demo")
    messaging_workflow = MessagingWorkflow(
        _created_at=datetime.datetime.now(),
        components=[comp, ],
        agents=[],
        product=product,
    )

    assert messaging_workflow.id is not None
    assert messaging_workflow.destination is None


def test_setup_messaging_workflow_with_provider():
    provider = ProductProvider(
        name="demo provider",
        region="US",
        data_pipeline=["data"],
        destinations=["email", "linkedin",]
    )
    product = Product(
        description="demo p",
        audience="demo audience",
        usp="demo usp",
        landing_url="www.com",
        cohort_timeframe=30,
        provider=provider,
    )
    comp = MessagingComponent(message="demo")
    messaging_workflow = MessagingWorkflow(
        _created_at=datetime.datetime.now(),
        components=[comp, ],
        agents=[],
        product=product,
    )

    assert messaging_workflow.id is not None
    assert messaging_workflow.destination is not None
    assert messaging_workflow.destination in provider.destinations
