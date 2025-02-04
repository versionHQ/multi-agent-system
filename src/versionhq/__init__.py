# silence some warnings
import warnings
warnings.filterwarnings(action="ignore", message="Pydantic serializer warnings:", category=UserWarning, module="pydantic.main")
warnings.filterwarnings(action="ignore", category=UserWarning, module="pydantic._internal")
warnings.filterwarnings(action="ignore", module="LiteLLM:utils")

from versionhq.agent.model import Agent
from versionhq.clients.customer.model import Customer
from versionhq.clients.product.model import Product, ProductProvider
from versionhq.clients.workflow.model import MessagingWorkflow, MessagingComponent
from versionhq.task.model import Task, TaskOutput
from versionhq.team.model import Team, TeamOutput
from versionhq.tool.model import Tool
from versionhq.tool.composio_tool import ComposioHandler


__version__ = "1.1.12.1"
__all__ = [
    "Agent",
    "Customer",
    "Product",
    "ProductProvider",
    "MessagingWorkflow",
    "MessagingComponent",
    "LLM",
    "Task",
    "TaskOutput",
    "Team",
    "TeamOutput",
    "Tool",
    "ComposioHandler"
]
