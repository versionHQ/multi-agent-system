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
        "openrouter/google/gemini-2.0-flash-thinking-exp:free",
        "openrouter/google/gemini-2.0-flash-thinking-exp-1219:free",
        "openrouter/google/gemini-2.0-flash-001",
        "openrouter/meta-llama/llama-3.3-70b-instruct",
        "openrouter/mistralai/mistral-large-2411",
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
    "gpt-4": 8192,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "o1-preview": 128000,
    "o1-mini": 128000,

    "gemini/gemini-1.5-flash": 1048576,
    "gemini/gemini-1.5-pro": 2097152,
    "gemini/gemini-2.0-flash-exp": 1048576,
    "gemini/gemini-gemma-2-9b-it": 8192,
    "gemini/gemini-gemma-2-27b-it": 8192,

    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-sonnet-20240620": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-opus-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    "claude-3-5-sonnet-2024102": 200000,

    "deepseek-chat": 128000,
    "deepseek/deepseek-reasoner": 8192,

    "gemma2-9b-it": 8192,
    "gemma-7b-it": 8192,
    "llama3-groq-70b-8192-tool-use-preview": 8192,
    "llama3-groq-8b-8192-tool-use-preview": 8192,
    "llama-3.1-70b-versatile": 131072,
    "llama-3.1-8b-instant": 131072,
    "llama-3.2-1b-preview": 8192,
    "llama-3.2-3b-preview": 8192,
    "llama-3.2-11b-text-preview": 8192,
    "llama-3.2-90b-text-preview": 8192,
    "llama3-70b-8192": 8192,
    "llama3-8b-8192": 8192,
    "mixtral-8x7b-32768": 32768,
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
        # "api_key",
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
