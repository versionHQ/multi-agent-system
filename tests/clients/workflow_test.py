import os
import pytest
import datetime

from versionhq.clients.workflow.model import Score, ScoreFormat, MessagingWorkflow, MessagingComponent
from versionhq.clients.product.model import Product, ProductProvider
from versionhq.tool.composio_tool_vars import ComposioAppName


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
    product = Product(description="demo p", audience="demo audience", usp="demo usp", landing_url="www.com")
    comp = MessagingComponent(message="demo")
    messaging_workflow = MessagingWorkflow(_created_at=datetime.datetime.now(), messaging_components=[comp, ], product=product)

    assert messaging_workflow.id is not None
    assert messaging_workflow.destination is None


def test_setup_messaging_workflow_with_provider():
    provider = ProductProvider(
        name="demo provider",
        region="US",
        data_pipelines=["data"],
        destination_services=["linkedin", "email",]
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
    messaging_workflow = MessagingWorkflow(_created_at=datetime.datetime.now(), messaging_components=[comp, ], product=product)

    assert messaging_workflow.id is not None
    assert messaging_workflow.destination is not None
    assert messaging_workflow.destination in provider.destination_services
    assert isinstance(messaging_workflow.destination, ComposioAppName)
