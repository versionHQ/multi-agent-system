import os
import pytest
import datetime

from versionhq.clients.workflow.model import Score, ScoreFormat, MessagingWorkflow, MessagingComponent
from versionhq.clients.product.model import Product, ProductProvider


def test_create_product():
    provider = ProductProvider()
    product = Product()
