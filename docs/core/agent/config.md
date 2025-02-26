## Optimizing Model

**Model Optimization**

`[var]`<bold>`llm: Optional[str | LLM | Dict[str, Any]] = "gpt-4o"`</bold>

You can select a model or model provider that the agent will run on.

By default, when the model provider name is provided, we will select the most cost-efficient model from the given provider.

```python
import versionhq as vhq

agent = vhq.Agent(role="Marketing Analyst", llm="gemini-2.0")
```

<hr/>


**Other LLM Configuration**

`[var]`<bold>`llm_config: Optional[Dict[str, Any]] = None`</bold>

You can specify any other parameters that the agent needs to follow when they call the LLM. Else, the agent will follow the default settings given by the model provider.

e.g. Expect longer context and form a short answer

```python
import versionhq as vhq

agent = vhq.Agent(
    role="Marketing Analyst",
    respect_context_window=False,
    max_execution_time=60,
    max_rpm=5,
    llm_config=dict(
            temperature=1,
            top_p=0.1,
            n=1,
            stop="answer",
            dummy="I am dummy" # <- invalid field will be ignored automatically.
        )
    )

assert isinstance(agent.llm, vhq.LLM)
assert agent.llm.llm_config["temperature"] == 1
assert agent.llm.llm_config["top_p"] == 0.1
assert agent.llm.llm_config["n"] == 1
assert agent.llm.llm_config["stop"] == "answer"
```

<hr>


## Building Knowledge

**Knowlege Source**

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

content = "Kuriko's favorite color is gold, and she enjoy Japanese food."
string_source = vhq.StringKnowledgeSource(content=content)

agent = vhq.Agent(
	role="Information Agent",
	goal="Provide information based on knowledge sources",
	knowledge_sources=[string_source,]
)

task = vhq.Task(
	description="Answer the following question: What is Kuriko's favorite color?"
)

res = task.execute(agent=agent)
assert "gold" in res.raw  == True
```

* Reference: <bold>`Knowledge` class</bold>

<hr />

## Accessing Memories

Store task execution results in memory

`[var]`<bold>`with_memory: bool = False`</bold>

By turning on the with_memory val True, the agent will create and store the task output and contextualize the memory when they execute the task.

```python
from versionhq.task.model import Agent

agent = vhq.Agent(
	role="Researcher",
	goal="You research about math.",
	with_memory=True
)

assert isinstance(agent.short_term_memory, vhq.ShortTermMemory)
assert isinstance(agent.long_term_memory, vhq.LongTermMemory)
```

<hr />

**RAG Storage**

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

## Updating Existing Agents

**Model configuration**

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

**Updating existing agents**

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
