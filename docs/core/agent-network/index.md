---
tags:
  - Agent Network
---

# Agent Network

<class>`class` versionhq.agent_network.model.<bold>AgentNetwork<bold></class>

A Pydantic class to store `AgentNetwork` objects that handle multiple agent formations for the task execution.

You can specify a desired formation or allow the agents to determine it autonomously (default).

|  | **Solo Agent** | **Supervising** | **Squad** | **Random** |
| :--- | :--- | :--- | :--- | :--- |
| **Formation** | <img src="https://res.cloudinary.com/dfeirxlea/image/upload/v1738818211/pj_m_agents/rbgxttfoeqqis1ettlfz.png" alt="solo" width="200"> | <img src="https://res.cloudinary.com/dfeirxlea/image/upload/v1738818211/pj_m_agents/zhungor3elxzer5dum10.png" alt="solo" width="200"> | <img src="https://res.cloudinary.com/dfeirxlea/image/upload/v1738818211/pj_m_agents/dnusl7iy7kiwkxwlpmg8.png" alt="solo" width="200"> | <img src="https://res.cloudinary.com/dfeirxlea/image/upload/v1738818211/pj_m_agents/sndpczatfzbrosxz9ama.png" alt="solo" width="200"> |
| **Usage** | <ul><li>A single agent with tools, knowledge, and memory.</li><li>When self-learning mode is on - it will turn into **Random** formation.</li></ul> | <ul><li>Leader agent gives directions, while sharing its knowledge and memory.</li><li>Subordinates can be solo agents or networks.</li></ul> | <ul><li>Share tasks, knowledge, and memory among network members.</li></ul> | <ul><li>A single agent handles tasks, asking help from other agents without sharing its memory or knowledge.</li></ul> |
| **Use case** | An email agent drafts promo message for the given audience. | The leader agent strategizes an outbound campaign plan and assigns components such as media mix or message creation to subordinate agents. | An email agent and social media agent share the product knowledge and deploy multi-channel outbound campaign. | 1. An email agent drafts promo message for the given audience, asking insights on tones from other email agents which oversee other clusters. 2. An agent calls the external agent to deploy the campaign. |

<hr>

## Quick Start

By default, lead agents will determine the best network formation autonomously based on the given task and its goal.

Calling `.launch()` method can start executing tasks and generate a tuple of response as a `TaskOutput` object and `TaskGraph` object.

```python
import versionhq as vhq

network = vhq.form_agent_network(
  task=f"create a promo plan to attract a client",
  expected_outcome='media mix, key messages, and CTA targets.'
)

res, tg = network.launch()

assert isinstance(res, vhq.TaskOutput)
assert isinstance(tg, vhq.TaskGraph)
```

Ref. <a href="/core/task-output">TaskOutput</a> / <a href="/core/task-graph">TaskGraph </a> class.

Visit <a href="https://versi0n.io">Playground</a>.
