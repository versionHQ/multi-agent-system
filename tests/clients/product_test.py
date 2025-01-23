import os
import pytest
import datetime

from versionhq.clients.workflow.model import Score, ScoreFormat, MessagingWorkflow, MessagingComponent
from versionhq.clients.product.model import Product, ProductProvider
from versionhq.tool.composio_tool_vars import ComposioAppName


def test_create_product_provider():
    provider = ProductProvider(name="demo", region="US", data_pipelines=["hubspot", "demo crm"] )

    assert provider.id is not None
    assert provider.name == "demo"
    assert isinstance(provider.data_pipelines[0], ComposioAppName)
    assert isinstance(provider.data_pipelines[1], str)
    assert provider.destination_services is None


def test_create_product():
    provider = ProductProvider(name="demo", region="US", data_pipelines=["hubspot", "demo crm"], destination_services=["test"])
    product = Product(name="demo", description="demo", provider=provider, audience="demo", usp="demo", landing_url="www.com", cohort_timeframe=30)

    assert product.id is not None
    assert product.name == "demo"
    assert product.provider.id is not None
