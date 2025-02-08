from typing import List, Dict, Any, Optional, Callable

from versionhq import Product, ProductProvider, Customer
from versionhq.clients.customer.model import  Status


def test_create_customer():
    provider = ProductProvider(name="demo", region="US", data_pipelines=["hubspot", "demo crm"], destination_services=["test"])
    product = Product(name="demo", description="demo", provider=provider, landing_url="www.com", cohort_timeframe=30)

    def custom_deploy(item):
        return f"custom deploy with {item}"

    class CustomCustomer(Customer):
        name: str = "custom"
        staus: Status = Status.READY_TO_DEPLOY
        products: List[Product] = [product,]
        analysis: str = "analysis"
        function: Optional[Callable] = custom_deploy
        config: Optional[Dict[str, Any]] = dict(item="custom config")

    customer = CustomCustomer()

    assert customer.id is not None
    assert customer.deploy() == "custom deploy with custom config"
    assert customer.status == Status.ACTIVE_ON_WORKFLOW
    assert [isinstance(item, Product) for item in customer.products]
