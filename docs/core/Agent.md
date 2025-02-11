---
tags:
  - HTML5
  - JavaScript
  - CSS
---

# Agent

<class>`class` versionhq.agent.model.<bold>Agent<bold></class>

Each agent has its unique knowledge and memory on the past task.

You can create one and assign the task or reassign another task to the existing agent after fine-tuning.


## Core usage

By defining its role and goal in a simple sentence, the AI agent will be set up to run on <bold>`gpt-4o`</bold> by default.

```python
import versionhq as vhq

agent = vhq.Agent(
	role="Marketing Analyst",
	goal="Coping with price competition in saturated markets"
)
```

<hr />

## Customization

### Model optimization

`[var]`<bold>`llm: Optional[str | LLM | Dict[str, Any]] = "gpt-4o"`</bold>

You can select a model or model provider that the agent will run on.

By default, when the model provider name is provided, we will select the most cost-efficient model from the given provider.

```python
import versionhq as vhq

agent = vhq.Agent(
	role="Marketing Analyst",
	goal="Coping with price competition in saturated markets",
	llm="gemini-2.0"
)
```

### Switching models

`[class method]`<bold>`update_llm(self, llm: Any = None, llm_config: Optional[Dict[str, Any]] = None) -> Self`<bold>

You can update LLM model and its configuration of the existing agent.

```python
import versionhq as vhq

agent = vhq.Agent(
	role="Marketing Analyst",
	goal="Coping with price competition in saturated markets",
	llm="gemini-2.0"
)

agent.update_llm(llm="deepseek", llm_config=dict(max_tokens=3000))
assert "deepseek-r1" in agent.llm.model
assert agent.llm.max_tokens == 3000
```

<hr/>

### Developer Prompt (System Prompt)

`[var]`<bold>`backstory: Optional[str] = TEMPLATE_BACKSTORY`<bold>

Backstory will be drafted automatically using the given role, goal and other values in the Agent model, and converted into the **developer prompt** when the agent executes the task.

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

print(agent.backstory)

# You are an expert marketing analyst with relevant skillsets and abilities to query relevant information from the given knowledge sources. Leveraging these, you will identify competitive solutions to achieve the following goal: coping with price competition in saturated markets.
```

You can also specify your own backstory by simply adding the value to the backstory field of the Agent model:

```python
import versionhq as vhq

agent = vhq.Agent(
	role="Marketing Analyst",
	goal="Coping with increased price competition in saturated markets.",
    backstory="You are a marketing analyst for a company in a saturated market. The market is becoming increasingly price-competitive, and your company's profit margins are shrinking. Your primary goal is to develop and implement strategies to help your company maintain its market share and profitability in this challenging environment."
)

print(agent.backstory)

# You are a marketing analyst for a company in a saturated market. The market is becoming increasingly price-competitive, and your company's profit margins are shrinking. Your primary goal is to develop and implement strategies to help your company maintain its market share and profitability in this challenging environment.
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

## Task Execution Rules

### Delegation

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

### Max Retry Limit

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

### Maximum Number of Iterations (MaxIt)

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

### Callbacks

`[var]`<bold>`callbacks: Optional[List[Callable]] = None`</bold>

You can add callback functions that the agent will run after executing any task.

By default, raw response from the agent will be added to the arguments of the callback function.

e.g. Format a response after executing the task:

```python
import json
import versionhq as vhq
from typing import Dict, Any


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
```

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

### Context Window

`[var]`<bold>`respect_context_window: [bool] = True`</bold>

A context window determines the amount of text that the model takes into account when generating a response.

By adjusting the context window, you can control the level of context the model considers while generating the output. A smaller context window focuses on immediate context, while a larger context window provides a broader context.

By default, the agent will follow **the 80% rule** - where they only use 80% of the context window limit of the LLM they run on.

You can turn off this rule by setting `respect_context_window` False to have larger context window.

### Max Tokens

`[var]`<bold>`max_tokens: Optional[int] = None`</bold>

Max tokens defines the maximum number of tokens in the generated response. Tokens can be thought of as the individual units of text, which can be words or characters.

By default, the agent will follow the default max_tokens of the model, but you can specify the max token to limit the length of the generated output.

### Maximum Execution Time

`[var]`<bold>`max_execution_times: Optional[int] = None`</bold>

The maximum amount of wall clock time to spend in the execution loop.

By default, the agent will follow the default setting of the model.

### Maximum RPM (Requests Per Minute)

`[var]`<bold>`max_rpm: Optional[int] = None`</bold>

The maximum number of requests that the agent can send to the LLM.

By default, the agent will follow the default setting of the model. When the value is given, we let the model sleep for 60 seconds when the number of executions exceeds the maximum requests per minute.

### Other LLM Configuration

`[var]`<bold>`llm_config: Optional[Dict[str, Any]] = None`</bold>

You can specify any other parameters that the agent needs to follow when they call the LLM. Else, the agent will follow the default settings given by the model provider.

e.g. Expect longer context and form a short answer

```python
import versionhq as vhq

agent = vhq.Agent(
    role="Marketing Analyst",
    goal="Coping with increased price competition in saturated markets.",
    respect_context_window=False,
    max_tokens=3000,
    max_execution_time=60,
    max_rpm=5,
    llm_config=dict(
            temperature=1,
            top_p=0.1,
            n=1,
            stream=False,
            stream_options=None,
            stop="test",
            max_completion_tokens=10000,
            dummy="I am dummy" # <- invalid field will be ignored automatically.
        )
    )

print(agent.llm)
# LLM(
#     max_tokens=3000,
#     temperature=1,
#     top_p=0.1,
#     n=1,
#     stream=False,
#     stream_options=None,
#     stop="test",
#     max_completion_tokens=10000,
# )
```

<hr />

## Knowledge

### Knowledge Sources

`[var]`<bold>`knowledge_sources: Optional[List[KnowledgeSource]] = None`</bold>

You can add knowledge sources to the agent in the following formats:

- Plane text
- Excel file
- PPTX
- PDF
- CSV
- JSON
- HTML file

The agent will run a query in the given knowledge source using the given context, then add the search results to the task prompt context.

```python
import versionhq as vhq
from versionhq.task.model import Task
from versionhq.knowledge.source import StringKnowledgeSource

content = "Kuriko's favorite color is gold, and she enjoy Japanese food."
string_source = StringKnowledgeSource(content=content)

agent = vhq.Agent(
	role="Information Agent",
	goal="Provide information based on knowledge sources",
	knowledge_sources=[string_source,]
)

task = Task(
	description="Answer the following question: What is Kuriko's favorite color?"
)

res = task.execute(agent=agent)
assert "gold" in res.raw  == True
```

* Reference: <bold>`Knowledge` class</bold>

<hr />

## Memory

### Store task execution results in memory

`[var]`<bold>`use_memory: bool = False`</bold>

By turning on the use_memory val True, the agent will create and store the task output and contextualize the memory when they execute the task.

```python
from versionhq.task.model import Agent

agent = vhq.Agent(
	role="Researcher",
	goal="You research about math.",
	use_memory=True
)

print(agent.short_term_memory)
# returns ShortTermMemory object.

print(agent.long_term_memory)
# returns LongTermMemory object.
```

### RAG Storage

When the agent is not given any `memory_config` values, they will create `RAGStorage` to store memory:

```python
RAGStorage(
	type="stm", # short-term memory
	allow_reset=True, # default = True. Explicitly mentioned.
	embedder_config=None,
	agents=[agent,]
)
```

MEM0 Storage

* Reference: <bold>`Memory`</bold> class

<hr />

## Utilities

### Model configuration

`[var]`<bold>`config: Optional[Dict[str, Any]] = None`</bold>

You can create an agent by using model config parameters instead.

e.g. Using config val

```python
import versionhq as vhq

agent = vhq.Agent(
	config=dict(
		role="Marketing Analyst",
		goal="Coping with increased price competition in saturated markets.",
	)
)
```

This is the same as the following:

```python
import versionhq as vhq

agent = vhq.Agent(
	role="Marketing Analyst",
	goal="Coping with price competition in saturated markets.",
)
```

<hr />

### Updating model values

`[class method]`<bold>`update(self, **kwargs) -> Self`</bold>

You can update values of exsiting agents using `update` class method.

This class method will safely trigger some setups that needs to be run before the agent start executing tasks.


```python
import versionhq as vhq

agent = vhq.Agent(
    role="Marketing Analyst",
    goal="Coping with price competition in saturated markets"
)

tool = vhq.Tool(func=lambda x: x)
agent.update(
    tools=[tool],
    goal="my new goal", # updating the goal (this will trigger updating the developer_prompt.)
    max_rpm=3,
    knowledge_sources=["testing", "testing2"], # adding knowledge sources (this will trigger the storage creation.)
    memory_config={"user_id": "0000"},
    llm="gemini-2.0", # Updating model (The valid llm_config for the new model will be inherited.)
    use_developer_prompt=False,
    dummy="I am dummy" # <- Invalid field will be automatically ignored.
)
```

<hr />

**List of models to run the agent**

```python
"openai": [
    "gpt-4",
    "gpt-4o",
    "gpt-4o-mini",
    "o1-mini",
    "o1-preview",
]

"gemini": [
    "gemini/gemini-1.5-flash",
    "gemini/gemini-1.5-pro",
    "gemini/gemini-2.0-flash-exp",
]

"anthropic": [
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-sonnet-20240229",
    "claude-3-opus-20240229",
    "claude-3-haiku-20240307",
]

"openrouter": [
    "openrouter/deepseek/deepseek-r1",
    "openrouter/qwen/qwen-2.5-72b-instruct",
]
```

Following models are under review.

```python
"huggingface": [
    "huggingface/qwen/qwen2.5-VL-72B-Instruct",
]

"sagemaker": [
    "sagemaker/huggingface-text2text-flan-t5-base",
    "sagemaker/huggingface-llm-gemma-7b",
    "sagemaker/jumpstart-dft-meta-textgeneration-llama-2-13b",
    "sagemaker/jumpstart-dft-meta-textgeneration-llama-2-70b",
    "sagemaker/jumpstart-dft-meta-textgeneration-llama-3-8b",
    "sagemaker/jumpstart-dft-meta-textgeneration-llama-3-70b",
    "sagemaker/huggingface-llm-mistral-7b"
]

"ollama": [
    "ollama/llama3.1",
    "ollama/mixtral",
    "ollama/mixtral-8x22B-Instruct-v0.1",
]

"watson": [
    "watsonx/meta-llama/llama-3-1-70b-instruct",
    "watsonx/meta-llama/llama-3-1-8b-instruct",
    "watsonx/meta-llama/llama-3-2-11b-vision-instruct",
    "watsonx/meta-llama/llama-3-2-1b-instruct",
    "watsonx/meta-llama/llama-3-2-90b-vision-instruct",
    "watsonx/meta-llama/llama-3-405b-instruct",
    "watsonx/mistral/mistral-large",
    "watsonx/ibm/granite-3-8b-instruct",
]

"bedrock": [
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    "bedrock/anthropic.claude-3-opus-20240229-v1:0",
    "bedrock/anthropic.claude-v2",
    "bedrock/anthropic.claude-instant-v1",
    "bedrock/meta.llama3-1-405b-instruct-v1:0",
    "bedrock/meta.llama3-1-70b-instruct-v1:0",
    "bedrock/meta.llama3-1-8b-instruct-v1:0",
    "bedrock/meta.llama3-70b-instruct-v1:0",
    "bedrock/meta.llama3-8b-instruct-v1:0",
    "bedrock/amazon.titan-text-lite-v1",
    "bedrock/amazon.titan-text-express-v1",
    "bedrock/cohere.command-text-v14",
    "bedrock/ai21.j2-mid-v1",
    "bedrock/ai21.j2-ultra-v1",
    "bedrock/ai21.jamba-instruct-v1:0",
    "bedrock/meta.llama2-13b-chat-v1",
    "bedrock/meta.llama2-70b-chat-v1",
    "bedrock/mistral.mistral-7b-instruct-v0:2",
    "bedrock/mistral.mixtral-8x7b-instruct-v0:1",
]
```
