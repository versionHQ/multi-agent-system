---
tags:
  - Agent Network
---

# Agent

<class>`class` versionhq.agent.model.<bold>Agent<bold></class>

A Pydantic class to store an `Agent` object that handles `Task` execution.


## Quick Start

By defining its role and goal in a simple sentence, the AI agent will be set up to run on <bold>`gpt-4o`</bold> by default.

Calling `.start()` method can start the agent operation, then generate response in text and JSON formats stored in the `TaskOutput` object.

```python
import versionhq as vhq

agent = vhq.Agent(
	role="Marketing Analyst",
	goal="Coping with price competition in saturated markets"
)

res = agent.start(context="Planning a new campaign promotion starting this summer")

assert agent.id
assert isinstance(res, vhq.TaskOutput)
assert res.json
```

Ref. <a href="/core/task">Task</a> class / <a href="/core/llm">LLM</a> class
