# Quick Start

## Package installation

```
pip install versionhq
```

(Python 3.11 | 3.12 | 3.13)

<hr />

## Forming agent networks

You can generate a network of multiple agents depending on your task complexity.

Here is a code snippet:

```python
import versionhq as vhq

network = vhq.form_agent_network(
   task="YOUR AMAZING TASK OVERVIEW",
   expected_outcome="YOUR OUTCOME EXPECTATION",
)
res, _ = network.launch()
```

This will form a network with multiple agents on `Formation` and return results as a `TaskOutput` object, storing outputs in JSON, plane text, Pydantic model formats along with evaluation.


## Building AI agents

If you don't need to form a network or assign a specific agent to the network, you can simply build an agent using `Agent` model.

Agents can execute tasks using `Task` model and return JSON format by default with plane text and pydantic model formats as options.


```python
import versionhq as vhq
from pydantic import BaseModel

class CustomOutput(BaseModel):
   test1: str
   test2: list[str]

def dummy_func(message: str, test1: str, test2: list[str]) -> str:
   return f"""{message}: {test1}, {", ".join(test2)}"""


agent = vhq.Agent(role="demo manager")

task = vhq.Task(
   description="Amazing task",
   pydantic_output=CustomOutput,
   callback=dummy_func,
   callback_kwargs=dict(message="Hi! Here is the result: ")
)

res = task.execute(agent=agent, context="amazing context to consider.")

assert isinstance(res, vhq.TaskOutput)
```

This will return a `TaskOutput` object that stores response in plane text, JSON, and Pydantic model: `CustomOutput` formats with a callback result, tool output (if given), and evaluation results (if given).

```python
res == TaskOutput(
   task_id=UUID('<TASK UUID>'),
   raw='{\"test1\":\"random str\", \"test2\":[\"str item 1\", \"str item 2\", \"str item 3\"]}',
   json_dict={'test1': 'random str', 'test2': ['str item 1', 'str item 2', 'str item 3']},
   pydantic=<class '__main__.CustomOutput'>,
   tool_output=None,
   callback_output='Hi! Here is the result: random str, str item 1, str item 2, str item 3', # returned a plain text summary
   evaluation=None
)
```

## Supervising

To create an agent network with one or more manager agents, designate members using the `is_manager` tag.

```python
import versionhq as vhq

agent_a = vhq.Agent(role="agent a", goal="My amazing goals", llm="llm-of-your-choice")
agent_b = vhq.Agent(role="agent b", goal="My amazing goals", llm="llm-of-your-choice")

task_1 = vhq.Task(
   description="Analyze the client's business model.",
   response_fields=[vhq.ResponseField(title="test1", data_type=str, required=True),],
   allow_delegation=True
)

task_2 = vhq.Task(
   description="Define a cohort.",
   response_fields=[vhq.ResponseField(title="test1", data_type=int, required=True),],
   allow_delegation=False
)

network =vhq.AgentNetwork(
   members=[
      vhq.Member(agent=agent_a, is_manager=False, tasks=[task_1]),
      vhq.Member(agent=agent_b, is_manager=True, tasks=[task_2]), # Agent B as a manager
   ],
)
res, _ = network.launch()

assert isinstance(res, vhq.NetworkOutput)
assert "agent b" in task_1.processed_agents # agent_b delegated by agent_a
assert "agent b" in task_2.processed_agents
```

This will return a list with dictionaries with keys defined in the `ResponseField` of each task.

Tasks can be delegated to a manager, peers within the agent network, or a completely new agent.
