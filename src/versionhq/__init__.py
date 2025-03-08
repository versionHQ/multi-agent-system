# silence some warnings
import warnings
warnings.filterwarnings(action="ignore", message="Pydantic serializer warnings:", category=UserWarning, module="pydantic.main")
warnings.filterwarnings(action="ignore", category=UserWarning, module="pydantic._internal")
warnings.filterwarnings(action="ignore", module="LiteLLM:utils")

from dotenv import load_dotenv
load_dotenv(override=True)

from versionhq.agent.model import Agent
from versionhq.agent_network.model import AgentNetwork, Formation, Member, TaskHandlingProcess
from versionhq.llm.model import LLM
from versionhq.llm.llm_vars import LLM_CONTEXT_WINDOW_SIZES, PARAMS, PROVIDERS, MODELS
from versionhq.clients.customer.model import Customer
from versionhq.clients.product.model import Product, ProductProvider
from versionhq.clients.workflow.model import MessagingWorkflow, MessagingComponent
from versionhq.knowledge.model import Knowledge, KnowledgeStorage
from versionhq.knowledge.source import PDFKnowledgeSource, CSVKnowledgeSource, JSONKnowledgeSource, TextFileKnowledgeSource, ExcelKnowledgeSource, StringKnowledgeSource
from versionhq.knowledge.source_docling import DoclingSource
from versionhq.task_graph.model import TaskStatus, TaskGraph, Node, Edge, DependencyType, Condition, ConditionType
from versionhq.task.model import Task, TaskOutput, ResponseField, TaskExecutionType
from versionhq.task.evaluation import Evaluation, EvaluationItem
from versionhq.tool.model import Tool, ToolSet
from versionhq.tool.rag_tool import RagTool
from versionhq.tool.cache_handler import CacheHandler
from versionhq.tool.tool_handler import ToolHandler
from versionhq.tool.composio_tool import ComposioHandler
from versionhq.memory.contextual_memory import ContextualMemory
from versionhq.memory.model import ShortTermMemory,LongTermMemory, UserMemory, MemoryItem

from versionhq.agent_network.formation import form_agent_network
from versionhq.task_graph.draft import workflow


__version__ = "1.2.4.1"
__all__ = [
    "Agent",

    "AgentNetwork",
    "Formation",
    "Member",
    "TaskHandlingProcess",

    "LLM",
    "LLM_CONTEXT_WINDOW_SIZES",
    "PARAMS",
    "PROVIDERS",
    "MODELS",

    "Customer",
    "Product",
    "ProductProvider",
    "MessagingWorkflow",
    "MessagingComponent",

    "Knowledge",
    "KnowledgeStorage",
    "PDFKnowledgeSource",
    "CSVKnowledgeSource",
    "JSONKnowledgeSource",
    "TextFileKnowledgeSource",
    "ExcelKnowledgeSource",
    "StringKnowledgeSource",
    "DoclingSource",

    "TaskStatus",
    "TaskGraph",
    "Node",
    "Edge",
    "DependencyType",
    "Condition",
    "ConditionType",

    "Task",
    "TaskOutput",
    "ResponseField",
    "TaskExecutionType",

    "Evaluation",
    "EvaluationItem",

    "Tool",
    "ToolSet",
    "RagTool",
    "CacheHandler",
    "ToolHandler",
    "ComposioHandler",

    "ContextualMemory",
    "ShortTermMemory",
    "LongTermMemory",
    "UserMemory",
    "MemoryItem",

    "form_agent_network",
    "workflow",
]
