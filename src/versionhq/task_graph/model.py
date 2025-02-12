import enum
import uuid
import networkx as nx
import matplotlib.pyplot as plt
from abc import ABC
from typing import List, Any, Optional, Callable, Dict, Type, Tuple

from pydantic import BaseModel, InstanceOf, Field, UUID4, field_validator
from pydantic_core import PydanticCustomError

from versionhq.task.model import Task, TaskOutput
from versionhq.agent.model import Agent
from versionhq._utils.logger import Logger


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



# class TriggerEvent(enum.Enum):
#     """
#     Concise enumeration of key trigger events for task execution.
#     """
#     IMMEDIATE = 0 # execute immediately
#     DEPENDENCIES_MET = 1  # All/required dependencies are satisfied
#     RESOURCES_AVAILABLE = 2  # Necessary resources are available
#     SCHEDULED_TIME = 3  # Scheduled start time or time window reached
#     EXTERNAL_EVENT = 4  # Triggered by an external event/message
#     DATA_AVAILABLE = 5  # Required data  is available both internal/external
#     APPROVAL_RECEIVED = 6  # Necessary approvals have been granted
#     STATUS_CHANGED = 7  # Relevant task/system status has changed
#     RULE_MET = 8  # A predefined rule or condition has been met
#     MANUAL_TRIGGER = 9  # Manually initiated by a user
#     ERROR_HANDLED = 10  # A previous error/exception has been handled



class Node(BaseModel):
    """
    A class to store a node object.
    """
    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    task: InstanceOf[Task] = Field(default=None)
    # trigger_event: TriggerEvent = Field(default=TriggerEvent.IMMEDIATE, description="store trigger event for starting the task execution")
    in_degree_nodes: List[Any] = Field(default_factory=list, description="list of Node objects")
    out_degree_nodes: List[Any] = Field(default_factory=list, description="list of Node objects")
    assigned_to: InstanceOf[Agent] = Field(default=None)
    status: TaskStatus = Field(default=TaskStatus.NOT_STARTED)


    @field_validator("id", mode="before")
    @classmethod
    def _deny_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError("may_not_set_field", "This field is not to be set by the user.", {})

    def is_independent(self) -> bool:
        return not self.in_degree_nodes and not self.out_degree_nodes


    def handle_task_execution(self, agent: Agent = None, context: str = None) -> TaskOutput | None:
        """
        Start task execution and update status accordingly.
        """

        self.status = TaskStatus.IN_PROGRESS

        if not self.task:
            Logger(verbose=True).log(level="error", message="Missing a task to execute. We'll return None.", color="red")
            self.status = TaskStatus.ERROR
            return None

        res = self.task.execute(agent=agent, context=context)
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
        """Unique identifier for the node"""
        return f"{str(self.id)}"

    def __str__(self):
        return self.identifier


class Edge(BaseModel):
    """
    A class to store an edge object that connects source and target nodes.
    """

    source: Node = Field(default=None)
    target: Node = Field(default=None)

    description: Optional[str] = Field(default=None)
    weight: Optional[float | int] = Field(default=1, description="est. duration for the task execution or respective weight of the target node (1 low - 10 high priority)")

    dependency_type: DependencyType = Field(default=DependencyType.FINISH_TO_START)
    required: bool = Field(default=True, description="whether to consider the source's status")
    condition: Optional[Callable] = Field(default=None, description="conditional function to start executing the dependency")
    condition_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict)

    lag: Optional[float | int] = Field(default=None, description="lag time (sec) from the dependency met to the task execution")
    data_transfer: bool = Field(default=True, description="whether the data transfer is required. by default transfer plane text output from in-degree nodes as context")


    def dependency_met(self) -> bool:
        """
        Defines if the dependency is ready to execute:

        required - condition - Dependency Met? - Dependent Task Can Start?
        True	Not Given	Predecessor task finished	Yes
        True	Given	Predecessor task finished and condition True	Yes
        False	Not Given	Always (regardless of predecessor status)	Yes
        False	Given	Condition True (predecessor status irrelevant)	Yes
        """

        if not self.required:
            return self.condition(**self.condition_kwargs) if self.condition else True

        match self.dependency_type:
            case DependencyType.FINISH_TO_START:
                """target starts after source finishes"""
                if self.source.status == TaskStatus.COMPLETED:
                    return self.condition(**self.conditon_kwargs) if self.condition else True
                else:
                    return False

            case DependencyType.START_TO_START:
                """target starts when source starts"""
                if self.source.status != TaskStatus.NOT_STARTED:
                    return self.condition(**self.conditon_kwargs) if self.condition else True
                else:
                    return False

            case DependencyType.FINISH_TO_FINISH:
                """target finish when source start"""
                if self.source.status != TaskStatus.COMPLETED:
                    return self.condition(**self.conditon_kwargs) if self.condition else True
                else:
                    return False

            case DependencyType.START_TO_FINISH:
                """target finishes when source start"""
                if self.source.status == TaskStatus.IN_PROGRESS:
                    return self.condition(**self.conditon_kwargs) if self.condition else True
                else:
                    return False


    def activate(self) -> TaskOutput | None:
        """
        Activates the edge to initiate task execution of the target node.
        """

        if not self.source or not self.target:
            Logger(verbose=True).log(level="warning", message="Cannot find source or target nodes. We'll return None.", color="yellow")
            return None

        if not self.dependency_met():
            Logger(verbose=True).log(level="warning", message="Dependencies not met. We'll see the source node status.", color="yellow")
            return None


        if self.lag:
            import time
            time.sleep(self.lag)

        context = self.source.task.output.raw if self.data_transfer else None
        res = self.target.handle_task_execution(context=context)
        return res


class Graph(ABC, BaseModel):
    """
    An abstract class to store G using NetworkX library.
    """

    directed: bool = Field(default=False, description="Whether the graph is directed")
    graph: Type[nx.Graph] = Field(default=None)
    nodes: Dict[str, InstanceOf[Node]] = Field(default_factory=dict, description="identifier: Node - for the sake of ")
    edges: Dict[str, InstanceOf[Edge]] = Field(default_factory=dict)

    def __init__(self, directed: bool = False, **kwargs):
        super().__init__(directed=directed, **kwargs)
        self.graph = nx.DiGraph() if self.directed else nx.Graph()

    def _return_node_object(self, node_identifier) -> Node | None:
        return [v for k, v in self.nodes.items() if k == node_identifier][0] if [v for k, v in self.nodes.items() if k == node_identifier] else None

    def add_node(self, node: Node) -> None:
        self.graph.add_node(node.identifier, **node.model_dump())
        self.nodes[node.identifier] = node

    def add_edge(self, source: str, target: str, edge: Edge) -> None:
        self.graph.add_edge(source, target, **edge.model_dump())
        edge.source = self._return_node_object(source)
        edge.source.out_degree_nodes.append(target)
        edge.target = self._return_node_object(target)
        edge.target.in_degree_nodes.append(source)
        self.edges[(source, target)] = edge

    def add_weighted_edges_from(self, edges):
        self.graph.add_weighted_edges_from(edges)

    def get_neighbors(self, node: Node) -> List[Node]:
        return list(self.graph.neighbors(node))

    def get_in_degree(self, node: Node) -> int:
        return self.graph.in_degree(node)

    def get_out_degree(self, node: Node) -> int:
        return self.graph.out_degree(node)

    def find_start_nodes(self) -> Tuple[Node]:
        return [v for k, v in self.nodes.items() if v.in_degrees == 0 and v.out_degrees > 0]

    def find_end_nodes(self) ->  Tuple[Node]:
        return [v for k, v in self.nodes.items() if v.out_degrees == 0 and v.in_degrees > 0]

    def find_critical_end_node(self) -> Node | None:
        """
        Find a critical end node from all the end nodes to lead a conclusion of the entire graph.
        """
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

        return list(critical_path), critical_duration, all_paths


    def is_circled(self, node: Node) -> bool:
        """Check if there's a path from the node to itself and return bool."""
        try:
            path = nx.shortest_path(self.graph, source=node, target=node)
            return True if path else False
        except nx.NetworkXNoPath:
            return False


    def visualize(self, title: str = "Task Graph", pos: Any = None, **graph_config):
        pos = pos if pos else nx.spring_layout(self.graph, seed=42)
        nx.draw(
            self.graph,
            pos,
            with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_color="black", arrowstyle='-|>', arrowsize=20, arrows=True,
            **graph_config
        )
        edge_labels = {}
        for u, v, data in self.graph.edges(data=True):
            edge = self.edges.get((u,v))
            if edge:
                label_parts = []
                if edge.type:
                    label_parts.append(f"Type: {edge.type}")
                if edge.duration is not None:
                    label_parts.append(f"Duration: {edge.duration}")
                if edge.lag is not None:
                    label_parts.append(f"Lag: {edge.lag}")
                edge_labels[(u, v)] = "\n".join(label_parts)  # Combine labels with newlines
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.title(title)
        plt.show()


class TaskGraph(Graph):
    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    should_reform: bool = Field(default=False)
    status: Dict[str, TaskStatus] = Field(default_factory=dict, description="store identifier (str) and TaskStatus of all task_nodes")
    outputs: Dict[str, TaskOutput] = Field(default_factory=dict, description="store node identifire and TaskOutput")
    conclusion: Any = Field(default=None, description="store the final result of the entire task graph. critical path target/end node")


    def _save(self, abs_file_path: str = None) -> None:
        """
        Save the graph image in the local directory.
        """

        try:
            import os
            project_root = os.path.abspath(os.getcwd())
            abs_file_path = abs_file_path if abs_file_path else f"{project_root}/uploads"

            os.makedirs(abs_file_path, exist_ok=True)
            plt.savefig(f"{abs_file_path}/{str(self.id)}.png")

        except Exception as e:
            Logger().log(level="error", message=f"Failed to save the graph {str(self.id)}: {str(e)}", color="red")


    def add_task(self, task: Node | Task) -> Node:
        """Convert `task` to a Node object and add it to G"""
        task_node = task if isinstance(task, Node) else Node(task=task)
        self.add_node(task_node)
        self.status[task_node.identifier] = TaskStatus.NOT_STARTED
        return task_node


    def add_dependency(
            self, source_task_node_identifier: str, target_task_node_identifier: str, **edge_attributes
        ) -> None:
        """
        Add an edge that connect task 1 (source) and task 2 (target) using task_node.name as an identifier
        """

        if not edge_attributes:
            Logger(verbose=True).log(level="error", message="Edge attributes are missing.", color="red")

        edge = Edge()
        for k in Edge.model_fields.keys():
            v = edge_attributes.get(k, None)
            if v:
                setattr(edge, k, v)
            else:
                pass

        self.add_edge(source_task_node_identifier, target_task_node_identifier, edge)


    def set_task_status(self, identifier: str, status: TaskStatus) -> None:
        if identifier in self.status:
            self.status[identifier] = status
        else:
            Logger().log(level="warning", message=f"Task '{identifier}' not found in the graph.", color="yellow")
            pass


    def get_task_status(self, identifier):
        if identifier in self.status:
            return self.status[identifier]
        else:
            Logger().log(level="warning", message=f"Task '{identifier}' not found in the graph.", color="yellow")
            return None


    def visualize(self, layout: str = None):
        try:
            pos = nx.drawing.nx_agraph.graphviz_layout(self.graph, prog='dot') # 'dot', 'neato', 'fdp', 'sfdp'
        except ImportError:
            pos = nx.spring_layout(self.graph, seed=42) # REFINEME - layout

        node_colors = list()
        for k, v in self.graph.nodes.items():
            status = self.get_task_status(identifier=k)
            if status == TaskStatus.NOT_STARTED:
                node_colors.append("skyblue")
            elif status == TaskStatus.IN_PROGRESS:
                node_colors.append("lightgreen")
            elif status == TaskStatus.BLOCKED:
                node_colors.append("lightcoral")
            elif status == TaskStatus.COMPLETED:
                node_colors.append("black")
            elif status == TaskStatus.DELAYED:
                node_colors.append("orange")
            elif status == TaskStatus.ON_HOLD:
                node_colors.append("yellow")
            else:
                node_colors.append("grey")

        critical_paths, duration, paths = self.find_critical_path()
        edge_colors = ['red' if (u, v) in zip(critical_paths, critical_paths[1:]) else 'black' for u, v in self.graph.edges()]
        edge_widths = []

        for k, v in self.edges.items():
            # edge_weights = nx.get_edge_attributes(self.graph, 'weight')
            # edge_colors.append(plt.cm.viridis(v.weight / max(edge_weights.values())))
            edge_widths.append(v.weight * 0.5)

        nx.draw(
            self.graph, pos,
            with_labels=True,
            node_size=700,
            node_color=node_colors,
            font_size=10,
            font_color="black",
            edge_color=edge_colors,
            width=edge_widths,
            arrows=True,
            arrowsize=20,
            arrowstyle='-|>'
        )

        edge_labels = nx.get_edge_attributes(G=self.graph, name="edges")
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        plt.title("Project Network Diagram")
        self._save()
        plt.show()


    def activate(self, target_node_identifier: Optional[str] = None) -> Tuple[TaskOutput | None, Dict[str, TaskOutput]]:
        """
        Starts to execute all nodes in the graph or a specific node if the target is given, following the given conditons of the edge obeject.
        Then returns tuple of the last task output and all task outputs (self.outputs)
        """
        if target_node_identifier:
            if not [k for k in self.nodes.keys() if k == target_node_identifier]:
                Logger().log(level="error", message=f"The node {str(target_node_identifier)} is not in the graph.", color="red")
                return None

            # find a shortest path to each in-degree node of the node and see if dependency met.
            node = self._return_node_object(target_node_identifier)
            sources = node.in_degrees
            edge_status = []
            res = None

            for item in sources:
                edge = self.find_path(source=item, target=target_node_identifier)
                edge_status.append(dict(edge=edge if edge else None, dep_met=edge.dependency_met() if edge else False))

            if len([item for item in edge_status if item["dep_met"] == True]) == len(sources):
                res = node.handle_task_execution()
                self.outputs.update({ target_node_identifier: res })
                self.status.update({ target_node_identifier: edge.target.status })

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

            # When all nodes are completed, return the output of the critical end node or end node.
            if end_nodes and len([node for node in end_nodes if node.status == TaskStatus.COMPLETED]) == len(end_nodes):
                if critical_end_node:
                    return critical_end_node.task.output, self.outputs

                else:
                    return [v.task.output.raw for k, v in end_nodes.items()][0], self.outputs

            # Else, execute nodes connected with the critical_path
            elif critical_path:
                for item in critical_path:
                    edge = [v for k, v in self.edges.items() if item in k]

                    if edge:
                        edge = edge[0]

                        if edge.target.status == TaskStatus.COMPLETED:
                            res = edge.target.output

                        else:
                            res = edge.activate()
                            node_identifier = edge.target.identifier
                            self.outputs.update({ node_identifier: res })
                            self.status.update({ node_identifier: edge.target.status })

                            if not res and start_nodes:
                                for node in start_nodes:
                                    res = node.handle_task_execution()
                                    self.outputs.update({ node.identifier: res })
                                    self.status.update({ node.identifier: node.status })

            # if no critical paths in the graph, simply start from the start nodes.
            elif start_nodes:
                for node in start_nodes:
                    res = node.handle_task_execution()
                    self.outputs.update({ node.identifier: res })
                    self.status.update({ node.identifier: node.status })


            # if none of above is applicable, try to activate all the edges.
            else:
                for k, edge in self.edges.items():
                    res = edge.activate()
                    node_identifier = edge.target.identifier
                    self.outputs.update({ node_identifier: res })
                    self.status.update({ node_identifier: edge.target.status })

            # last_task_output = [v for v in self.outputs.values()][len([v for v in self.outputs.values()]) - 1] if [v for v in self.outputs.values()] else None
            return res, self.outputs
