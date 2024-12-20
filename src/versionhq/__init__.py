import warnings
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:",
    category=UserWarning,
    module="pydantic.main",
)

from versionhq.agent.model import Agent
from versionhq.clients.customer.model import Customer
from versionhq.clients.product.model import Product
from versionhq.clients.workflow.model import MessagingWorkflow
from versionhq.task.model import Task
from versionhq.team.model import Team
from versionhq.tool.model import Tool


__version__ = "1.1.0"
__all__ = [
    "Agent",
    "Customer",
    "Product",
    "MessagingWorkflow",
    "Task",
    "Team",
    "Tool",
]
