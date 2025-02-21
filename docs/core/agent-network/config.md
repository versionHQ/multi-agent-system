---
tags:
  - Agent Network
---


## Adding Members

<class>`class` versionhq.agent_network.model.<bold>Member<bold></class>

You can simply add an agent as a member using `members` field.

```python
import versionhq as vhq

network = vhq.AgentNetwork(
    members=[
        vhq.Member(
            agent=vhq.Agent(role="new member", goal="work in the network"),
            is_manager=False  # explicitly mentioned. Setting `True` makes this member a manager of the network.
        ),
    ]
)
assert isinstance(network.members[0].agent, vhq.Agent)
```

Ref. <a href="/core/agent-network/ref/#member">`Member`</a> class.

<hr />

## Changing Formation

The formation of network members will be automatically assigned based on the task goals, but you can explicitly define it by using `formation` field.


```python
import versionhq as vhq

network = vhq.AgentNetwork(
  members=[
    vhq.Member(agent=vhq.Agent(role="member 1", goal="work in the network")),
    vhq.Member(agent=vhq.Agent(role="member 2", goal="work in the network")),
    vhq.Member(agent=vhq.Agent(role="member 3", goal="work in the network")),
  ],
  formation=vhq.Formation.SQUAD,
)

assert network.formation == vhq.Formation.SQUAD
```

Ref. Enum <a href="/core/agent-network/ref/#enum-formation">`Formation`</a>


<hr >

## Task Handling

The class method `.launch()` will automatically decide the best task handling process and execute the tasks accordingly.


```python
import versionhq as vhq

network = vhq.AgentNetwork(
  members=[
      vhq.Member(agent=vhq.Agent(role="member 1", goal="work in the network"), tasks=[vhq.Task(description="Run a demo 1")]),
      vhq.Member(agent=vhq.Agent(role="member 2", goal="work in the network"), tasks=[vhq.Task(description="Run a demo")]),
      vhq.Member(agent=vhq.Agent(role="member 3", goal="work in the network")),
  ],
)

res, tg = network.launch()

assert isinstance(res, vhq.TaskOutput)
assert isinstance(tg, vhq.TaskGraph)

```

<hr >

You can also specify the process using `process` field.

```python
import versionhq as vhq

network = vhq.AgentNetwork(
  members=[
      vhq.Member(agent=vhq.Agent(role="member 1", goal="work in the network"), tasks=[vhq.Task(description="Run a demo 1")]),
      vhq.Member(agent=vhq.Agent(role="member 2", goal="work in the network"), tasks=[vhq.Task(description="Run a demo 2")]),
      vhq.Member(agent=vhq.Agent(role="member 3", goal="work in the network")),
  ],
  process=vhq.TaskHandlingProcess.CONSENSUAL,
  consent_trigger=lambda x: True, # consent trigger event is a MUST for TaskHandlingProcess.CONSENSUAL
)

res, tg = network.launch()

assert isinstance(res, vhq.TaskOutput)
assert isinstance(tg, vhq.TaskGraph)
```

Ref. Enum <a href="/core/agent-network/ref/#enum-taskhandlingprocess">`TaskHandlingProcess`</a>
