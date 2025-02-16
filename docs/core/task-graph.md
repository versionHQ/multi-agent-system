---
tags:
  - Task Graph
---

# TaskGraph

<class>`class` versionhq.graph.model.<bold>TaskGraph<bold></class>

A `TaskGraph` represents tasks as `nodes` and their execution dependencies as `edges`, automating rule-based execution.

`Agent Networks` can handle `TaskGraph` objects by optimizing their formations.

The following example demonstrates a simple concept of a `supervising` agent network handling a task graph with three tasks and one critical edge.

<img src="https://res.cloudinary.com/dfeirxlea/image/upload/v1739337639/pj_m_home/zfg4ccw1m1ww1tpnb0pa.png">


## Quick Start

`TaskGraph` needs at least two `nodes` and one `edges` to connect the nodes to function.

You can define nodes and edges mannually by creating nodes from tasks, and defining edges.


### Generating

```python
import versionhq as vhq

task_graph = vhq.TaskGraph(directed=False, should_reform=True)

task_a = vhq.Task(description="Research Topic")
task_b = vhq.Task(description="Outline Post")
task_c = vhq.Task(description="Write First Draft")

node_a = task_graph.add_task(task=task_a)
node_b = task_graph.add_task(task=task_b)
node_c = task_graph.add_task(task=task_c)

task_graph.add_dependency(
    node_a.identifier, node_b.identifier,
    dependency_type=vhq.DependencyType.FINISH_TO_START, weight=5, description="B depends on A"
)
task_graph.add_dependency(
    node_a.identifier, node_c.identifier,
    dependency_type=vhq.DependencyType.FINISH_TO_FINISH, lag=1, required=False, weight=3
)

critical_path, duration, paths = task_graph.find_critical_path()

import uuid
assert isinstance(task_graph, vhq.TaskGraph)
assert [type(k) == uuid.uuid4 and isinstance(v, vhq.Node) and isinstance(v.task, vhq.Task) for k, v in task_graph.nodes.items()]
assert [type(k) == uuid.uuid4 and isinstance(v, vhq.Edge) for k, v in task_graph.edges.items()]
assert critical_path  and duration  and paths
```


### Activating

Calling `.activate()` begins execution of the graph's nodes, respecting dependencies [`dependency-met`] and prioritizing the critical path.


**[NOTES]**

- If all nodes are already complete, outputs are returned without further execution.

- If no critical path is found, execution begins with any dependency-met start nodes.


```python
import versionhq as vhq

# Inherting the `task_graph` object in the previous code snippet,

last_task_output, outputs = task_graph.activate()

assert isinstance(last_task_output, vhq.TaskOutput)
assert [k in task_graph.nodes.keys() and v and isinstance(v, vhq.TaskOutput) for k, v in outputs.items()]
```
