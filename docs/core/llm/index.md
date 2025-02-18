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

Some add-on features are unavailable for the followings models and providers:

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
