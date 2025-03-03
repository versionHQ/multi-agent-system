---
tags:
  - Utilities
---


# RAG Tool

<class>`class` versionhq.tool.rag_tool.<bold>RagTool<bold></class>

A Pydantic class to store RAG tools that the agent will use when it executes the task.


## Quick Start

Similar to the `Tool` class, you can run the RAG tool using `url` and `query` variables.

```python
import versionhq as vhq

rt = vhq.RagTool(
    url="https://github.com/chroma-core/chroma/issues/3233",
    query="What is the next action plan?"
)
res = rt.run()

assert rt.text is not None # text source from the url
assert res is not None
```


<hr>

## Using with Agents

You can call a specific agent when you run a RAG tool.

```python
import versionhq as vhq

rt = vhq.RagTool(url="https://github.com/chroma-core/chroma/issues/3233", query="What is the next action plan?")

agent = vhq.Agent(role="RAG Tool Tester")
res = rt.run(agent=agent)

assert agent.knowledge_sources is not None
assert rt.text is not None
assert res is not None
```


Agents can own RAG tools.

```python
import versionhq as vhq

rt = vhq.RagTool(url="https://github.com/chroma-core/chroma/issues/3233", query="What is the next action plan?")

agent = vhq.Agent(role="RAG Tool Tester", tools=[rt]) # adding RAG tool/s
task = vhq.Task(description="return a simple response", can_use_agent_tools=True, tool_res_as_final=True)
res = task.execute(agent=agent)

assert res.raw is not None
assert res.tool_output is not None
```


### Variables

| <div style="width:160px">**Variable**</div> | **Data Type** | **Default** | **Nullable** | **Description** |
| :---               | :---  | :--- | :--- | :--- |
| **`api_key_name`** | Optional[str]   | None | True | API key name in .env file. |
| **`api_endpoint`**       | Optional[str]   | None | True |API endpoint. |
| **`url`** | Optional[str] | None | True | URLs to extract the text source. |
| **`headers`** | Optional[Dict[str, Any]]  | dict() | - | Request headers |
| **`query`** |  Optional[str] | None | True | Query. |
| **`text`** |  Optional[str] | None | True | Text sources extracted from the URL or API call |


### Class Methods

| <div style="width:120px">**Method**</div> |  <div style="width:300px">**Params**</div> | **Returns** | **Description** |
| :---               | :---  | :--- | :--- |
| **`store_data`**  | <p>agent: Optional["vhq.Agent"] = Non</p> | None | Stores the retrieved data in the storage. |
| **`run`**  | *args, **kwargs | Any | Execute the tool. |
