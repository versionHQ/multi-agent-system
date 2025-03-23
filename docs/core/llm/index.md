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
    "gpt-4",
    "gpt-4o",
    "gpt-4o-mini",
    "o3-mini",
    "o3-mini-2025-01-31",
    "o1-mini",
    "o1-preview",
],
"gemini": [
    "gemini/gemini-2.0-flash-exp",
    "gemini/gemini-2.0-flash",
    "gemini/gemini-2.0-flash-thinking-exp",
    "gemini/gemini-2.0-flash-lite-preview-02-05",
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
    "bedrock/us.meta.llama3-2-90b-instruct-v1:0",
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
"azure": [
    "azure/DeepSeek-V3",
    "azure/DeepSeek-R1",
    "azure/Llama-3.3-70B-Instruct",
    "azure/Llama-3.2-11B-Vision-Instruct",
    "azure/Meta-Llama-3.1-405B-Instruct",
    "azure/Meta-Llama-3.1-8B-Instruct",
    "azure/Llama-3.2-1B-Instruct",
    "azure/Meta-Llama-3.1-70B",
    "azure/Meta-Llama-3.1-8B",
    "azure/Llama-3.2-3B-Instruct",
    "azure/Meta-Llama-3-8B-Instruct",
    "azure/Meta-Llama-3.1-70B-Instruct",
    "azure/Llama-3.2-90B-Vision-Instruct",
    "azure/Llama-3.2-3B",
    "azure/Llama-3.2-1B",
    "azure/mistral-large-latest",
    "azure/mistral-large-2402",
    "azure/command-r-plus",
    "azure/o3-mini-2025-01-31",
    "azure/o3-mini",
    "azure/o1-mini",
    "azure/Phi-4-mini-instruct",
    "azure/Phi-4-multimodal-instruct",
    "azure/Mistral-Large-2411",
    "azure/Mistral-small"
    "azure/mistral-small-2503",
    "azure/Ministral-3B",
    "azure/mistralai-Mixtral-8x22B-v0-1"
    "azure/Cohere-rerank-v3.5",
],
"azure_ai": [
    "azure_ai/DeepSeek-V3",
    "azure_ai/DeepSeek-R1",
    "azure_ai/Llama-3.3-70B-Instruct",
    "azure_ai/Llama-3.2-11B-Vision-Instruct",
    "azure_ai/Meta-Llama-3.1-405B-Instruct",
    "azure_ai/Meta-Llama-3.1-8B-Instruct",
    "azure_ai/Llama-3.2-1B-Instruct",
    "azure_ai/Meta-Llama-3.1-70B",
    "azure_ai/Meta-Llama-3.1-8B",
    "azure_ai/Llama-3.2-3B-Instruct",
    "azure_ai/Meta-Llama-3-8B-Instruct",
    "azure_ai/Meta-Llama-3.1-70B-Instruct",
    "azure_ai/Llama-3.2-90B-Vision-Instruct",
    "azure_ai/Llama-3.2-3B",
    "azure_ai/Llama-3.2-1B",
    "azure_ai/mistral-large-latest",
    "azure_ai/mistral-large-2402",
    "azure_ai/command-r-plus",
    "azure_ai/o3-mini-2025-01-31",
    "azure_ai/o3-mini",
    "azure_ai/o1-mini",
    "azure_ai/Phi-4-mini-instruct",
    "azure_ai/Phi-4-multimodal-instruct",
    "azure_ai/Mistral-Large-2411",
    "azure_ai/Mistral-small"
    "azure_ai/mistral-small-2503",
    "azure_ai/Ministral-3B",
    "azure_ai/mistralai-Mixtral-8x22B-v0-1"
    "azure_ai/Cohere-rerank-v3.5",
],
"huggingface": [
    "huggingface/qwen/qwen2.5-VL-72B-Instruct",
]
```
