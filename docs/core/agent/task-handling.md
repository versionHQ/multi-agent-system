
## Prompt Engineering

**Developer Prompt**

`[var]`<bold>`backstory: Optional[str] = TEMPLATE_BACKSTORY`<bold>

Backstory will be drafted automatically using the given role, goal and other values in the Agent model, and converted into the **developer prompt** when the agent executes the task.

<hr/>

**Backstory template (full) for auto drafting:**

```python
BACKSTORY_FULL="""You are an expert {role} highly skilled in {skills}. You have abilities to query relevant information from the given knowledge sources and use tools such as {tools}. Leveraging these, you will identify competitive solutions to achieve the following goal: {goal}."""
```

For example, the following agent’s backstory will be auto drafted using a simple template.

```python
import versionhq as vhq

agent = vhq.Agent(
	role="Marketing Analyst",
	goal="Coping with price competition in saturated markets"
)

assert agent.backstory == "You are an expert marketing analyst with relevant skills and abilities to query relevant information from the given knowledge sources. Leveraging these, you will identify competitive solutions to achieve the following goal: coping with price competition in saturated markets."
```

You can also specify your own backstory by simply adding the value to the backstory field of the Agent model:

```python
import versionhq as vhq

agent = vhq.Agent(
	role="Marketing Analyst",
	goal="Coping with increased price competition in saturated markets.",
    backstory="You are a marketing analyst for a company in a saturated market. The market is becoming increasingly price-competitive, and your company's profit margins are shrinking. Your primary goal is to develop and implement strategies to help your company maintain its market share and profitability in this challenging environment."
)

assert agent.backstory == "You are a marketing analyst for a company in a saturated market. The market is becoming increasingly price-competitive, and your company's profit margins are shrinking. Your primary goal is to develop and implement strategies to help your company maintain its market share and profitability in this challenging environment."
```
<hr />

`[var]`<bold>`use_developer_prompt: [bool] = True`</bold>

You can turn off the system prompt by setting `use_developer_prompt` False. In this case, the backstory is ignored when the agent call the LLM.

```python
import versionhq as vhq

agent = vhq.Agent(
	role="Marketing Analyst",
	goal="Coping with increased price competition in saturated markets.",
    use_developer_prompt=False # default - True
)
```

<hr >

## Executing Tasks

**Delegation**

`[var]`<bold>`allow_delegation: [bool] = False`</bold>

When the agent is occupied with other tasks or not capable enough to the given task, you can delegate the task to another agent or ask another agent for additional information. The delegated agent will be selected based on nature of the given task and/or tool.

```python
import versionhq as vhq

agent = vhq.Agent(
	role="Marketing Analyst",
	goal="Coping with increased price competition in saturated markets.",
	allow_delegation=True
)
```

<hr />

**Max Retry Limit**

`[var]`<bold>`max_retry_limit: Optional[int] = 2`</bold>

You can define how many times the agent can retry the execution under the same given conditions when it encounters an error.

```python
import versionhq as vhq

agent = vhq.Agent(
	role="Marketing Analyst",
	goal="Coping with increased price competition in saturated markets.",
	max_retry_limit=3
)
```

<hr />

**Maximum Number of Iterations (maxit)**

`[var]`<bold>`maxit: Optional[int] = 25`</bold>

You can also define the number of loops that the agent will run after it encounters an error.

i.e., The agent will stop the task execution after the 30th loop.

```python
import versionhq as vhq

agent = vhq.Agent(
	role="Marketing Analyst",
	goal="Coping with increased price competition in saturated markets.",
    maxit=30 # default = 25
)
```

<hr />

**Context Window**

`[var]`<bold>`respect_context_window: [bool] = True`</bold>

A context window determines the amount of text that the model takes into account when generating a response.

By adjusting the context window, you can control the level of context the model considers while generating the output. A smaller context window focuses on immediate context, while a larger context window provides a broader context.

By default, the agent will follow **the 80% rule** - where they only use 80% of the context window limit of the LLM they run on.

You can turn off this rule by setting `respect_context_window` False to have larger context window.


<hr />

**Max Tokens**

`[var]`<bold>`max_tokens: Optional[int] = None`</bold>

Max tokens defines the maximum number of tokens in the generated response. Tokens can be thought of as the individual units of text, which can be words or characters.

By default, the agent will follow the default max_tokens of the model, but you can specify the max token to limit the length of the generated output.


<hr />

**Maximum Execution Time**

`[var]`<bold>`max_execution_times: Optional[int] = None`</bold>

The maximum amount of wall clock time to spend in the execution loop.

By default, the agent will follow the default setting of the model.


<hr />

**Maximum RPM (Requests Per Minute)**

`[var]`<bold>`max_rpm: Optional[int] = None`</bold>

The maximum number of requests that the agent can send to the LLM.

By default, the agent will follow the default setting of the model. When the value is given, we let the model sleep for 60 seconds when the number of executions exceeds the maximum requests per minute.

<hr />


## Callbacks

`[var]`<bold>`callbacks: Optional[List[Callable]] = None`</bold>

You can add callback functions that the agent will run after executing any task.

By default, raw response from the agent will be added to the arguments of the callback function.

e.g. Format a response after executing the task:

```python
import json
from typing import Dict, Any

import versionhq as vhq

def format_response(res: str = None) -> str | Dict[str, Any]:
	try:
		r = json.dumps(eval(res))
		formatted_res = json.loads(r)
		return formatted_res
	except:
		return res

agent = vhq.Agent(
	role="Marketing Analyst",
	goal="Coping with increased price competition in saturated markets.",
	callbacks=[format_response]
)
res = agent.start()

assert res.callback_output
```

<hr />

**Multiple callbacks to call**

The callback functions are called in order of the list index referring to the task response and response from the previous callback functions by default.

e.g. Validate an initial response from the assigned agent, and format the response.

```python
import json
from typing import Dict, Any

import versionhq as vhq

def assessment(res: str) -> str:
    try:
        sub_agent = vhq.Agent(role="Validator", goal="Validate the given solutions.")
        task = vhq.Task(
            description=f"Assess the given solution based on feasibilities and fits to client's strategies, then refine the solution if necessary.\nSolution: {res}"
        )
        r = task.sync_execute(agent=sub_agent)
        return r.raw

    except:
        return res

def format_response(res: str = None) -> str | Dict[str, Any]:
    try:
        r = json.dumps(eval(res))
        formatted_res = json.loads(r)
        return formatted_res
    except:
        return res

agent = vhq.Agent(
    role="Marketing Analyst",
    goal="Build solutions to address increased price competition in saturated markets",
    callbacks=[assessment, format_response] # add multiple funcs as callbacks - executed in order of index
)
```
