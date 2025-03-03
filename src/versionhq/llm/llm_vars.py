from typing import Type

JSON_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

PROVIDERS = [
    "openai",
    "gemini",
    "openrouter",
    "anthropic",
    "bedrock",
    "bedrock/converse",
    "huggingface",
]

ENDPOINT_PROVIDERS = [
    "huggingface",
]


MODELS = {
    "openai": [
        "gpt-4.5-preview-2025-02-27",
        "gpt-4",
        "gpt-4o",
        "gpt-4o-mini",
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
    "huggingface": [
        "huggingface/qwen/qwen2.5-VL-72B-Instruct",
    ],
}


ENV_VARS = {
    "openai": ["OPENAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "huggingface": ["HUGGINGFACE_API_KEY", ],
    "bedrock": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"],
    "sagemaker": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"],
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

    "gemini/gemini-1.5-flash": 1048576,
    "gemini/gemini-1.5-pro": 2097152,
    "gemini/gemini-2.0-flash-exp": 1048576,

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
Params for litellm.completion().
"""

PARAMS = {
    "litellm": [
        "api_base",
        "api_version,",
        "num_retries",
        "context_window_fallback_dict",
        "fallbacks",
        "metadata",
        "api_key",
    ],
    "common": [
        "model",
        "messages",
        "temperature",
        "top_p",
        "max_tokens",
        "stream",
        "tools",
        "tool_choice",
        "response_format",
        "n",
        "stop",
        # "base_url",
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
