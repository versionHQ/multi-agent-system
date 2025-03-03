---
tags:
  - Task Graph
---

# Task

<class>`class` versionhq.task.model.<bold>Task<bold></class>

A Pydantic class to store and manage information for individual tasks, including their assignment to agents or agent networks, and dependencies via a node-based system that tracks conditions and status.

Ref. Node / Edge / <a href="/core/task-graph">TaskGraph</a> class

<hr />

## Quick Start

Create a task by defining its description in one simple sentence. The `description` will be used in the prompt later.

Each task will be assigned a unique ID as an identifier.

```python
import versionhq as vhq

task = vhq.Task(description="MY AMAZING TASK")

import uuid
assert uuid.UUID(str(task.id), version=4)
```


And you can simply execute the task by calling `.execute()` function.

```python
import versionhq as vhq

task = vhq.Task(description="MY AMAZING TASK")
res = task.execute()

assert isinstance(res, vhq.TaskOutput) # Generates TaskOutput object
assert res.raw and res.json # By default, TaskOutput object stores output in plane text and json formats.
assert task.processed_agents is not None # Agents will be automatically assigned to the given task.
```

<hr />

## Evaluating

`[var]`<bold>`should_evaluate: bool = False`</bold>

`[var]`<bold>`eval_criteria: Optional[List[str]] = list()`</bold>

You can turn on customized evaluations using the given criteria.

Refer <a href="/core/task/task-output">TaskOutput</a> class for details.
