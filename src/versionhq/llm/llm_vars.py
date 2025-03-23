from typing import Type

JSON_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

PROVIDERS = {
    "openai": { "api_key": "OPENAI_API_KEY" },
    "gemini": { "api_key": "GEMINI_API_KEY" },
    "openrouter": { "api_key": "OPENROUTER_API_KEY" },
    "anthropic": { "api_key": "ANTHROPIC_API_KEY" },
    "bedrock":  {
        "aws_access_key_id": "AWS_ACCESS_KEY_ID",
        "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
        "aws_region_name": "AWS_REGION_NAME",
    },
    "bedrock/converse":  {
        "aws_access_key_id": "AWS_ACCESS_KEY_ID",
        "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
        "aws_region_name": "AWS_REGION_NAME",
    },
    "sagemaker":  {
        "aws_access_key_id": "AWS_ACCESS_KEY_ID",
        "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
        "aws_region_name": "AWS_REGION_NAME",
    },
    "huggingface": {
        "api_key": "HUGGINGFACE_API_KEY",
        "base_url": "HUGGINGFACE_API_BASE",
        "HF_ENDPOINT": "HF_ENDPOINT",
    },
    "azure":  {
        "api_base": "AZURE_OPENAI_ENDPOINT",
        "api_key": "AZURE_OPENAI_API_KEY",
        "api_version": "AZURE_OPENAI_API_VERSION",
    },
    "azure_ai": {
        "api_key": "AZURE_AI_API_KEY",
        "base_url": "AZURE_AI_API_BASE",

    }
}

ENDPOINTS = [
    "azure", # endpoints must be aligned with the selected model.
    "azure_ai", # endpoints must be aligned with the selected model.
    "huggingface",
]


# Resaoning and text generation models
TEXT_MODELS = {
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
        "gemini/gemini-2.0-flash",
        "gemini/gemini-2.0-flash-thinking-exp",
        "gemini/gemini-2.0-flash-lite-preview-02-05",
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
    ],
}


"""
Max input token size by the model.
"""
LLM_CONTEXT_WINDOW_SIZES = {
    "gpt-4.5-preview-2025-02-27": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4": 8192,
    "o1-preview": 128000,
    "o1-mini": 128000,
    "o3-mini": 200000,
    "o3-mini-2025-01-31": 200000,

    "gemini/gemini-2.0-flash-exp": 1048576,
    "gemini/gemini-2.0-flash": 1048576,
    "gemini/gemini-2.0-flash-thinking-exp": 1048576,
    "gemini/gemini-2.0-flash-lite-preview-02-05": 1048576,

    "claude-3-7-sonnet-latest": 200000,
    "claude-3-5-haiku-latest": 200000,
    "claude-3-5-sonnet-latest": 200000,
    "claude-3-opus-latest": 200000,

    "openrouter/deepseek/deepseek-r1" :65336,
    "openrouter/qwen/qwen-2.5-72b-instruct": 33792,
    "openrouter/google/gemini-2.0-flash-001": 1048576,
    "openrouter/mistralai/mistral-large": 32000,
    "openrouter/cohere/command-r-plus": 128000,
    "openrouter/databricks/dbrx-instruct": 32768,

    "bedrock/converse/us.meta.llama3-3-70b-instruct-v1:0": 8192,
    "bedrock/us.meta.llama3-2-1b-instruct-v1:0": 128000,
    "bedrock/us.meta.llama3-2-3b-instruct-v1:0": 128000,
    "bedrock/us.meta.llama3-2-11b-instruct-v1:0": 128000,
    "bedrock/us.meta.llama3-2-90b-instruct-v1:0": 128000,
    "bedrock/mistral.mistral-7b-instruct-v0:2": 8191,
    "bedrock/mistral.mixtral-8x7b-instruct-v0:1": 8191,
    "bedrock/mistral.mistral-large-2407-v1:0": 8191,
    "bedrock/amazon.titan-text-lite-v1": 4000,
    "bedrock/amazon.titan-text-express-v1": 8000,
    "bedrock/amazon.titan-text-premier-v1:0": 32000,
    "bedrock/cohere.command-r-plus-v1:0": 4096,
    "bedrock/cohere.command-r-v1:0": 4096,
    "bedrock/cohere.command-text-v14": 4096,
    "bedrock/cohere.command-light-text-v14": 4096,
}



"""
Model config related params for litellm.completion().
"""
MODEL_PARAMS = {
    "litellm": [
        "num_retries",
        "context_window_fallback_dict",
        "fallbacks",
        "metadata",
    ],
    "common": [
        # "model",
        # "messages",
        # "stream",
        "response_format",
        "temperature",
        "top_p",
        "max_tokens",
        "tools",
        "tool_choice",
        "n",
        "stop",
    ],
    "openai": [
        "timeout",
        "stream_options",
        "max_compl,etion_tokens",
        "modalities",
        "prediction",
        "audio",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "seed",
        "logprobs",
        "top_logprobs",
        "parallel_tool_calls",
        "extra_headers",
        "model_list"
    ],
    "gemini": [
        "topK",
    ],
    "bedrock": {
        "top-k",
    }
}


class SchemaType:
    """
    A class to store/convert a LLM-valid schema type from the Python Type object.
    https://swagger.io/docs/specification/v3_0/data-models/data-types/
    https://cloud.google.com/vertex-ai/docs/reference/rest/v1/Schema#Type
    """

    def __init__(self, type: Type):
        self.type = type

    def convert(self) -> str:
        if self.type is None:
            return "string"

        if self.type is int:
            return "integer"
        elif self.type is float:
            return "number"
        elif self.type is str:
            return "string"
        elif self.type is dict:
            return "object"
        elif self.type is list:
            return "array"
        elif self.type is bool:
            return "boolean"
        else:
            return "string"
