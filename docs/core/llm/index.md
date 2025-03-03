---
tags:
  - Agent Network
---

# LLM

<class>`class` versionhq.llm.model.<bold>LLM<bold></class>

A Pydantic class to store LLM objects and its task handling rules.

You can specify a model and integration platform from the list. Else, we'll use `gemini` or `gpt` via `LiteLLM` by default.


**List of available models**

```python
"openai": [
    "gpt-4.5-preview-2025-02-27",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4",
    "o1-mini",
    "o1-preview",
],
"gemini": [
    "gemini/gemini-1.5-flash",
    "gemini/gemini-1.5-pro",
    "gemini/gemini-2.0-flash-exp",
],
"anthropic": [
    "claude-3-7-sonnet-latest",
    "claude-3-5-haiku-latest",
    "claude-3-5-sonnet-latest",
    "claude-3-opus-latest",
],
"openrouter": [
    "openrouter/deepseek/deepseek-r1",
    "openrouter/qwen/qwen-2.5-72b-instruct",
    "openrouter/google/gemini-2.0-flash-001",
    "openrouter/mistralai/mistral-large",
    "openrouter/cohere/command-r-plus",
    "openrouter/databricks/dbrx-instruct",
],
"bedrock": [
    "bedrock/converse/us.meta.llama3-3-70b-instruct-v1:0",
    "bedrock/us.meta.llama3-2-1b-instruct-v1:0",
    "bedrock/us.meta.llama3-2-3b-instruct-v1:0",
    "bedrock/us.meta.llama3-2-11b-instruct-v1:0",
    "bedrock/mistral.mistral-7b-instruct-v0:2",
    "bedrock/mistral.mixtral-8x7b-instruct-v0:1",
    "bedrock/mistral.mistral-large-2407-v1:0",
    "bedrock/amazon.titan-text-lite-v1",
    "bedrock/amazon.titan-text-express-v1",
    "bedrock/amazon.titan-text-premier-v1:0",
    "bedrock/cohere.command-r-plus-v1:0",
    "bedrock/cohere.command-r-v1:0",
    "bedrock/cohere.command-text-v14",
    "bedrock/cohere.command-light-text-v14",
],
"huggingface": [
    "huggingface/qwen/qwen2.5-VL-72B-Instruct",
]
```
