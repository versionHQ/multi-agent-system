---
tags:
  - Agent Network
---


You can generate an `AgentNetwork` by using `form_agent_network` method with a concise `task` description and `expected_outcome` args.

```python
import versionhq as vhq

network = vhq.form_agent_network(
    task="Find the best trip destination this summer.",
    expected_outcome="a list of destinations and why it's suitable",
    context="planning a suprise trip for my friend", # optional
)

assert isinstance(network, vhq.AgentNetwork)
assert network.members # auto-generated agents as network members
assert network.tasks # auto-defined sub-tasks to achieve the main task goal
```


<hr>

**Strucured Output**

To generate structured output, you can add a JSON dict or Pydantic class as `expected_outcome` args instead of plane text.

```python
import versionhq as vhq
from pydantic import BaseModel

class Outcome(BaseModel):
    destinations: list[str]
    why_suitable: list[str]


network = vhq.form_agent_network(
    task="Find the best trip destination this summer.",
    expected_outcome=Outcome,
    context="planning a suprise trip for my friend", # optional
)

assert isinstance(network, vhq.AgentNetwork)
assert network.members
assert network.tasks
```

<hr >

**Agents**

You can use `agents` args to add existing agents to the network.

```python
import versionhq as vhq
from pydantic import BaseModel

my_agent = vhq.Agent(
    role="Travel Agent",
    goal="select best trip destination",
    knowledge_sources=[".....","url1",]
)

class Outcome(BaseModel):
    destinations: list[str]
    why_suitable: list[str]

network = vhq.form_agent_network(
    task="Find the best trip destination this summer.",
    expected_outcome=Outcome,
    context="planning a suprise trip for my friend",
    agents=[my_agent,]
)

assert isinstance(network, vhq.AgentNetwork)
assert [member for member in network.members if member.agent == my_agent]
assert network.tasks
```

<hr>

**Formation**

Similar to `agents`, you can define `formation` args to specify the network formation:


```python
import versionhq as vhq
from pydantic import BaseModel

my_agent = vhq.Agent(
    role="Travel Agent",
    goal="select best trip destination",
    knowledge_sources=[".....","url1",]
)

class Outcome(BaseModel):
    destinations: list[str]
    why_suitable: list[str]

network = vhq.form_agent_network(
    task="Find the best trip destination this summer.",
    expected_outcome=Outcome,
    context="planning a suprise trip for my friend",
    agents=[my_agent,],
    formation=vhq.Formation.SUPERVISING
)

assert isinstance(network, vhq.AgentNetwork)
assert [member for member in network.members if member.agent == my_agent]
assert network.tasks
assert network.formation == vhq.Formation.SUPERVISING
```

Ref. Enum <a href="core/agent-network/ref/#enum-formation">Formation</a>
