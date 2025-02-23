import matplotlib
matplotlib.use('agg')

import enum
import uuid
import networkx as nx
import matplotlib.pyplot as plt
from abc import ABC
from concurrent.futures import Future
from typing import List, Any, Optional, Callable, Dict, Type, Tuple
from typing_extensions import Self

from pydantic import BaseModel, InstanceOf, Field, UUID4, field_validator, model_validator
from pydantic_core import PydanticCustomError

from versionhq.agent.model import Agent
from versionhq.task.model import Task, TaskOutput, Evaluation
from versionhq._utils.logger import Logger

class ConditionType(enum.Enum):
    AND = 1
    OR = 2


class Condition(BaseModel):
    """
    A Pydantic class to store edge conditions and their args and types.
    """
    # edge_id: UUID4 = uuid.uuid4()
    methods: Dict[str, Callable | "Condition"] = dict()
    args: Dict[str, Dict[str, Any]] = dict()
    type: ConditionType = None

    @model_validator(mode="after")
    def validate_type(self) -> Self:
        if len(self.methods.keys()) > 1 and self.type is None:
            raise PydanticCustomError("missing_type", "Missing type", {})
        return self

    def _execute_method(self, key: str, method: Callable | "Condition") -> bool:
        match method:
            case Condition():
                return method.condition_met()
            case _:
                args = self.args[key] if key in self.args else None
                res = method(**args) if args else method()
                return res


    def condition_met(self) -> bool:
        if not self.methods:
            return True

        if len(self.methods) == 1:
            for k, v in self.methods.items():
               return self._execute_method(key=k, method=v)

        else:
            cond_list = []
            for k, v in self.methods.items():
                res = self._execute_method(key=k, method=v)
                if self.type == ConditionType.OR and res == True:
                    return True
                elif self.type == ConditionType.AND and res == False:
                    return False
            return bool(len([item for item in cond_list if item == True]) == len(cond_list))


class TaskStatus(enum.Enum):
    """
    Enum to track the task execution status
    """
    NOT_STARTED = 1
    IN_PROGRESS = 2
    WAITING = 3     # waiting for its dependant tasks to complete. resumption set as AUTO.
    COMPLETED = 4
    DELAYED = 5     # task in progress - but taking longer than expected duration
    ON_HOLD = 6     # intentionally paused due to external factors and decisions. resumption set as DECISION.
    ERROR = 7       # tried task execute but returned error. resupmtion follows edge weights and agent settings


class DependencyType(enum.Enum):
    """
    Concise enumeration of the edge type.
    """
    FINISH_TO_START = "FS"  # Task B starts after Task A finishes
    START_TO_START = "SS"  # Task B starts when Task A starts
    FINISH_TO_FINISH = "FF"  # Task B finishes when Task A finishes
    START_TO_FINISH = "SF"  # Task B finishes when Task A starts


class Node(BaseModel):
    """
    A class to store a node object.
    """

    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    task: InstanceOf[Task] = Field(default=None)
    in_degree_nodes: List[Any] = Field(default_factory=list, description="list of Node objects")
    out_degree_nodes: List[Any] = Field(default_factory=list, description="list of Node objects")
    assigned_to: InstanceOf[Agent] = Field(default=None)
    status: TaskStatus = Field(default=TaskStatus.NOT_STARTED)

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError("may_not_set_field", "This field is not to be set by client.", {})


    def is_independent(self) -> bool:
        return not self.in_degree_nodes and not self.out_degree_nodes

    def handle_task_execution(self, agent: Agent = None, context: str = None, response_format: Type[BaseModel] = None) -> TaskOutput | None:
        """Executes the task and updates its status"""

        self.status = TaskStatus.IN_PROGRESS

        if not self.task:
            Logger().log(level="error", message="Missing a task to execute. We'll return None.", color="red")
            self.status = TaskStatus.ERROR
            return None

        agent = agent if agent else self.assigned_to
        self.task.pydantic_output = self.task.pydantic_output if self.task.pydantic_output else response_format if type(response_format) == BaseModel else None
        res = self.task.execute(agent=agent, context=context)

        if isinstance(res, Future): # activate async
            res = res.result()

        self.status = TaskStatus.COMPLETED if res else TaskStatus.ERROR
        return res

    @property
    def in_degrees(self) -> int:
        return len(self.in_degree_nodes) if self.in_degree_nodes else 0

    @property
    def out_degrees(self) -> int:
        return len(self.out_degree_nodes) if self.out_degree_nodes else 0

    @property
    def degrees(self) -> int:
        return self.in_degrees + self.out_degrees

    @property
    def identifier(self) -> str:
        """Unique identifier of the node"""
        return f"{str(self.id)}"

    @property
    def label(self) -> str:
        """Human friendly label for visualization"""
        return self.task.name if self.task.name else self.task.description[0: 8]

    def __str__(self):
        if self.task:
            return f"{self.identifier}: {self.task.name if self.task.name else self.task.description[0: 12]}"
        else:
            return self.identifier


class Edge(BaseModel):
    """
    A Pydantic class to store an edge object that connects source and target nodes.
    """

    source: Node = Field(default=None)
    target: Node = Field(default=None)

    description: Optional[str] = Field(default=None)
    weight: Optional[float | int] = Field(default=1, description="est. duration of the task execution or respective weight of the target node at any scale i.e., 1 low - 10 high priority")
    dependency_type: DependencyType = Field(default=DependencyType.FINISH_TO_START)
    required: bool = Field(default=True, description="whether to consider the source's status")
    condition: Optional[Condition] = Field(default=None)

    lag: Optional[float | int] = Field(default=None, description="lag time (sec) from the dependency met to the task execution")
    data_transfer: bool = Field(default=True, description="whether the data transfer is required. by default transfer plane text output from in-degree nodes as context")


    def _response_schema(self) -> Type[BaseModel]:
        class EdgeResponseSchema(BaseModel):
            wight: float
            dependecy_type: str
            required: bool
            need_condition: bool
            lag_in_sec: float

        return EdgeResponseSchema


    def dependency_met(self) -> bool:
        """
        Defines if the dependency is ready to execute:

        required - condition - Dependency Met? - Dependent Task Can Start?
        True	Not Given	Predecessor task finished	Yes
        True	Given	Predecessor task finished and condition True	Yes
        False	Not Given	Always (regardless of predecessor status)	Yes
        False	Given	Condition True (predecessor status irrelevant)	Yes
        """

        if self.required == False:
            return self.condition.condition_met() if self.condition else True
            # return self.condition(**self.condition_kwargs) if self.condition else True

        match self.dependency_type:
            case DependencyType.FINISH_TO_START:
                """target starts after source finishes"""
                if not self.source or self.source.status == TaskStatus.COMPLETED:
                    return self.condition.condition_met() if self.condition else True
                else:
                    return False

            case DependencyType.START_TO_START:
                """target starts when source starts"""
                if not self.source or self.source.status != TaskStatus.NOT_STARTED:
                    return self.condition.condition_met() if self.condition else True
                else:
                    return False

            case DependencyType.FINISH_TO_FINISH:
                """target finish when source start"""
                if not self.source or self.source.status != TaskStatus.COMPLETED:
                    return self.condition.condition_met() if self.condition else True
                else:
                    return False

            case DependencyType.START_TO_FINISH:
                """target finishes when source start"""
                if not self.source or self.source.status == TaskStatus.IN_PROGRESS:
                    return self.condition.condition_met() if self.condition else True
                else:
                    return False


    def activate(self, response_format: Type[BaseModel] = None) -> TaskOutput | None:
        """
        Activates the edge to initiate task execution of the target node.
        """

        if not self.source or not self.target:
            Logger(verbose=True).log(level="warning", message="Cannot find source or target nodes. We'll return None.", color="yellow")
            return None

        if not self.dependency_met():
            Logger(verbose=True).log(level="warning", message="Dependencies not met. We'll return None.", color="yellow")
            return None

        if self.lag:
            import time
            time.sleep(self.lag)

        context = self.source.task.output.raw if self.data_transfer else None
        res = self.target.handle_task_execution(context=context, response_format=response_format)
        return res

    @property
    def label(self):
        """Human friendly label for visualization."""
        return f"e{self.source.label}-{self.target.label}"


class Graph(ABC, BaseModel):
    """
    An abstract class to store G using NetworkX library.
    """
    directed: bool = Field(default=False, description="Whether the graph is directed")
    graph: Type[nx.Graph] = Field(default=None)
    nodes: Dict[str, Node] = Field(default_factory=dict, description="{node_identifier: Node}")
    edges: Dict[str, Edge] = Field(default_factory=dict)

    def __init__(self, directed: bool = False, **kwargs):
        super().__init__(directed=directed, **kwargs)
        self.graph = nx.DiGraph(directed=True) if self.directed else nx.Graph()

    def _return_node_object(self, node_identifier) -> Node | None:
        match = [v for k, v in self.nodes.items() if k == node_identifier]

        if match:
            node = match[0] if isinstance(match[0], Node) else match[0]["node"] if "node" in match[0] else None
            return node
        else:
            return None

    def add_node(self, node: Node) -> None:
        if node.identifier in self.nodes.keys():
            return
        self.graph.add_node(node.identifier, node=node)
        self.nodes[node.identifier] = node

    def add_edge(self, source: str, target: str, edge: Edge) -> None:
        self.graph.add_edge(source, target, edge=edge)

        source_node, target_node = self._return_node_object(source), self._return_node_object(target)

        edge.source = source_node
        source_node.out_degree_nodes.append(target_node)

        edge.target = target_node
        target_node.in_degree_nodes.append(source_node)

        self.edges[(source, target)] = edge

    def add_weighted_edges_from(self, edges):
        self.graph.add_weighted_edges_from(edges)

    def get_neighbors(self, node: Node) -> List[Node]:
        return list(self.graph.neighbors(node))

    def get_in_degree(self, node: Node) -> int:
        return self.graph.in_degree(node)

    def get_out_degree(self, node: Node) -> int:
        return self.graph.out_degree(node)

    def find_start_nodes(self) -> List[Node]:
        return [v for v in self.nodes.values() if v.in_degrees == 0 and v.out_degrees > 0]

    def find_end_nodes(self) ->  List[Node]:
        return [v for v in self.nodes.values() if v.out_degrees == 0 and v.in_degrees > 0]

    def find_critical_end_node(self) -> Node | None:
        """Finds a critical end node from all the end nodes to lead a conclusion of the entire graph."""
        end_nodes = self.find_end_nodes()
        if not end_nodes:
            return None

        if len(end_nodes) == 1:
            return end_nodes[0]

        edges = [v for k, v in self.edges if isinstance(v, Edge) and v.source in end_nodes]
        critical_edge = max(edges, key=lambda item: item['weight']) if edges else None
        return critical_edge.target if critical_edge else None

    def find_path(self, source: Optional[str] | None, target: str, weight: Optional[Any] | None) -> Any:
        try:
            return nx.shortest_path(self.graph, source=source, target=target, weight=weight)
        except nx.NetworkXNoPath:
            return None

    def find_all_paths(self,  source: str, target: str) -> List[Any]:
        return list(nx.all_simple_paths(self.graph, source=source, target=target))

    def find_critical_path(self) -> tuple[List[Any], int, Dict[str, int]]:
        """
        Finds the critical path in the graph.
        Returns:
            A tuple containing:
                - The critical path (a list of edge identifiers).
                - The duration of the critical path.
                - A dictionary of all paths and their durations.
        """

        all_paths = {}
        for start_node in self.find_start_nodes():
            for end_node in self.find_end_nodes(): # End at nodes with 0 out-degree
                for edge in nx.all_simple_paths(self.graph, source=start_node.identifier, target=end_node.identifier):
                    edge_weight = sum(self.edges.get(item).weight if self.edges.get(item) else 0 for item in edge)
                    all_paths[tuple(edge)] = edge_weight

        if not all_paths:
            return [], 0, all_paths

        critical_path = max(all_paths, key=all_paths.get)
        critical_duration = all_paths[critical_path]

        return list(set(critical_path)), critical_duration, all_paths

    def is_circled(self, node: Node) -> bool:
        """Check if there's a path from the node to itself and return bool."""
        try:
            path = nx.shortest_path(self.graph, source=node, target=node)
            return True if path else False
        except nx.NetworkXNoPath:
            return False


class TaskGraph(Graph):
    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    should_reform: bool = Field(default=False)
    outputs: Dict[str, TaskOutput] = Field(default_factory=dict, description="stores node identifier and TaskOutput")
    concl_template: Optional[Dict[str, Any] | Type[BaseModel]] = Field(default=None, description="stores final response format in Pydantic class or JSON dict")
    concl: Optional[TaskOutput] = Field(default=None, description="stores the final or latest conclusion of the entire task graph")


    def _save(self, title: str, abs_file_path: str = None) -> None:
        """
        Save the graph image in the local directory.
        """
        try:
            import os
            project_root = os.path.abspath(os.getcwd())
            abs_file_path = abs_file_path if abs_file_path else f"{project_root}/.diagrams"
            title = title if title else f"vhq-Diagram-{str(self.id)}"

            os.makedirs(abs_file_path, exist_ok=True)
            plt.savefig(f"{abs_file_path}/{title}.png")

        except Exception as e:
            Logger().log(level="error", message=f"Failed to save the graph {str(self.id)}: {str(e)}", color="red")


    def add_task(self, task: Node | Task) -> Node:
        """Convert `task` to a Node object and add it to G"""

        if isinstance(task, Node) and task.identifier in self.nodes.keys():
            return task

        elif isinstance(task, Task):
            match = []
            for v in self.nodes.values():
                if type(v) == dict and v["node"] and v["node"].task == task:
                    match.append(v["node"])
                elif v.task == task:
                    match.append(v)

            if match:
                return match[0]
            else:
                node = Node(task=task)
                self.add_node(node)
                return node

        else:
            task_node = task if isinstance(task, Node) else Node(task=task)
            self.add_node(task_node)
            return task_node


    def add_dependency(self, source: str, target: str, **edge_attributes) -> None:
        """
        Add an edge that connect task 1 (source) and task 2 (target) using task_node.name as an identifier
        """

        if not edge_attributes:
            Logger().log(level="error", message="Edge attributes are missing.", color="red")

        edge = Edge()
        for k in Edge.model_fields.keys():
            v = edge_attributes.get(k, None)
            if v is not None:
                setattr(edge, k, v)
            else:
                pass

        self.add_edge(source, target, edge)


    def get_task_status(self, identifier: str) -> TaskStatus | None:
        """Retrieves the latest status of the given node"""
        if not identifier or identifier not in self.nodes.keys():
            Logger().log(level="error", message=f"Task node: {identifier} is not in the graph.", color="red")
            return None

        return self._return_node_object(identifier).status


    def visualize(self, layout: str = None, should_save: bool = False):
        from matplotlib.lines import Line2D
        from versionhq.task_graph.colors import white, black, darkgrey, grey, primary, orange, lightgreen, green, darkgreen, darkergreen

        try:
            pos = nx.drawing.nx_agraph.graphviz_layout(self.graph, prog='dot') # 'dot', 'neato', 'fdp', 'sfdp'
        except ImportError:
            pos = nx.spring_layout(self.graph, seed=42) # REFINEME - layout

        node_colors, legend_elements = list(), list()
        for k, v in self.nodes.items():
            status = self.get_task_status(identifier=k)
            node =  v if isinstance(v, Node) else v["node"] if hasattr(v, "node") else None
            if node:
                task_name = node.task.name if node.task and node.task.name else node.task.description if node.task else None
                legend_label = f"ID {str(node.identifier)[0: 6]}...: {task_name}" if task_name else f"ID {str(node.identifier)[0: 6]}..."
                match status:
                    case TaskStatus.NOT_STARTED:
                        node_colors.append(green)
                        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=legend_label, markerfacecolor=green))

                    case TaskStatus.IN_PROGRESS:
                        node_colors.append(darkgreen)
                        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=legend_label, markerfacecolor=darkgreen))

                    case TaskStatus.WAITING:
                        node_colors.append(lightgreen)
                        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=legend_label, markerfacecolor=lightgreen))

                    case TaskStatus.COMPLETED:
                        node_colors.append(darkergreen)
                        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=legend_label, markerfacecolor=darkergreen))

                    case TaskStatus.DELAYED:
                        node_colors.append(orange)
                        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=legend_label, markerfacecolor=orange))

                    case TaskStatus.ON_HOLD:
                        node_colors.append(grey)
                        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=legend_label, markerfacecolor=grey))

                    case TaskStatus.ERROR:
                        node_colors.append(primary)
                        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=legend_label, markerfacecolor=primary))

                    case _:
                        node_colors.append(white)
                        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=legend_label, markerfacecolor=white))

        critical_paths, duration, paths = self.find_critical_path()
        edge_colors = [black if (u, v) in zip(critical_paths, critical_paths[1:]) else darkgrey for u, v in self.graph.edges()]
        edge_widths = [v.weight * 0.5 for v in self.edges.values()]

        # for k, v in self.edges.items():
        #     # edge_weights = nx.get_edge_attributes(self.graph, 'weight')
        #     # edge_colors.append(plt.cm.viridis(v.weight / max(edge_weights.values())))
        #     edge_widths.append(v.weight * 0.5)

        nx.draw(
            self.graph, pos,
            with_labels=True,
            node_size=600,
            node_color=node_colors,
            font_size=8,
            font_color=darkgrey,
            edge_color=edge_colors,
            width=edge_widths,
            arrows=True,
            arrowsize=15,
            arrowstyle='-|>'
        )

        edge_labels = nx.get_edge_attributes(G=self.graph, name="edges")
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        plt.legend(handles=legend_elements, loc='lower right')
        plt.title(f"vhq-Diagram {str(self.id)}")

        if should_save:
            self._save(title=f"vhq-Diagram {str(self.id)}")
        plt.show(block=False)


    def activate(self, target: Optional[str] = None) -> Tuple[TaskOutput | None, Dict[str, TaskOutput]]:
        """
        Starts to execute all nodes in the graph or a specific node if the target is given, following the given conditons of the edge obeject.
        Then returns tuple of the last task output and all task outputs (self.outputs)
        """

        Logger().log(color="blue", message=f"Start to activate the graph: {str(self.id)}", level="info")

        if target:
            if not [k for k in self.nodes.keys() if k == target]:
                Logger().log(level="error", message=f"The node {str(target)} is not in the graph.", color="red")
                return None, None

            # find a shortest path to each in-degree node of the node and see if dependency met.
            node = self._return_node_object(target)
            sources = node.in_degrees
            edge_status = []
            res = None

            for item in sources:
                edge = self.find_path(source=item, target=target)
                edge_status.append(dict(edge=edge if edge else None, dep_met=edge.dependency_met() if edge else False))

            if len([item for item in edge_status if item["dep_met"] == True]) == len(sources):
                res = node.handle_task_execution()
                self.outputs.update({ target: res })

            return res, self.outputs

        else:
            if not self.edges or not self.nodes:
                Logger().log(level="error", message="TaskGraph needs at least 2 nodes and 1 edge to activate. We'll return None.", color="red")
                return None

            start_nodes = self.find_start_nodes()
            end_nodes = self.find_end_nodes()
            critical_end_node = self.find_critical_end_node()
            critical_path, _, _ = self.find_critical_path()
            res = None

            if end_nodes and len([node for node in end_nodes if node.status == TaskStatus.COMPLETED]) == len(end_nodes):
                res = self.concl if self.concl else critical_end_node.task.output if critical_end_node else  [v.task.output for v in end_nodes.values()][0]

            elif critical_path:
                nodes = [v for k, v in self.nodes.items() if k in critical_path]
                if nodes:
                    for node in nodes:
                        if node.status == TaskStatus.COMPLETED:
                            res = node.task.output

                        else:
                            res = node.handle_task_execution()
                            self.outputs.update({ node.identifier: res })

                            if not res and start_nodes:
                                for node in start_nodes:
                                    res = node.handle_task_execution()
                                    self.outputs.update({ node.identifier: res })
                else:
                    for k, edge in self.edges.items():
                        res = edge.activate()
                        node_identifier = edge.target.identifier
                        self.outputs.update({ node_identifier: res })

            elif start_nodes:
                for node in start_nodes:
                    res = node.handle_task_execution()
                    self.outputs.update({ node.identifier: res })

            else:
                for k, edge in self.edges.items():
                    res = edge.activate()
                    node_identifier = edge.target.identifier
                    self.outputs.update({ node_identifier: res })

            self.concl = res
            self.concl_template = self.concl_template if self.concl_template else res.pydantic.__class__ if res.pydantic else None
             # last_task_output = [v for v in self.outputs.values()][len([v for v in self.outputs.values()]) - 1] if [v for v in self.outputs.values()] else None
            return res, self.outputs


    def evaluate(self, eval_criteria: List[str] = None) -> Evaluation | None:
        """Evaluates the conclusion based on the given eval criteria."""

        if not isinstance(self.concl, TaskOutput):
            return None

        tasks = [v.task for v in self.nodes.values() if v.task and v.task.id == self.concl.task_id]
        task = tasks[0] if tasks else None

        if not task:
            return None

        if not task.eval_criteria:
            task.eval_criteria = eval_criteria

        eval = self.concl.evaluate(task=task)
        return eval


    @property
    def usage(self) -> Tuple[int, float]:
        """Returns aggregate number of consumed tokens and job latency in ms during the activation"""

        tokens, latency = 0, 0
        for v in self.outputs.values():
            tokens += v._tokens
            latency += v.latency

        return tokens, latency
