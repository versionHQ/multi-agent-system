from enum import Enum
from typing import Type

JSON_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"


"""
List of models available on the framework.
Model names align with the LiteLLM's key names defined in the JSON URL.
Provider names align with the custom provider or model provider names.
-> model_key = custom_provider_name/model_name

Option
litellm.pick_cheapest_chat_models_from_llm_provider(custom_llm_provider: str, n=1)
"""

MODELS = {
    "openai": [
        # "gpt-3.5-turbo",
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
        # "gemini/gemini-gemma-2-9b-it",
        # "gemini/gemini-gemma-2-27b-it",
    ],
    # "vetrex_ai": [
    # ],
    "anthropic": [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-sonnet-20240229",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
    ],
    # "ollama": [
    #   "ollama/llama3.1",
    #   "ollama/mixtral",
    # ],
    # "watson": [
    #     "watsonx/meta-llama/llama-3-1-70b-instruct",
    #     "watsonx/meta-llama/llama-3-1-8b-instruct",
    #     "watsonx/meta-llama/llama-3-2-11b-vision-instruct",
    #     "watsonx/meta-llama/llama-3-2-1b-instruct",
    #     "watsonx/meta-llama/llama-3-2-90b-vision-instruct",
    #     "watsonx/meta-llama/llama-3-405b-instruct",
    #     "watsonx/mistral/mistral-large",
    #     "watsonx/ibm/granite-3-8b-instruct",
    # ],
    # "bedrock": [
    #     "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    #     "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    #     "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    #     "bedrock/anthropic.claude-3-opus-20240229-v1:0",
    #     "bedrock/anthropic.claude-v2:1",
    #     "bedrock/anthropic.claude-v2",
    #     "bedrock/anthropic.claude-instant-v1",
    #     "bedrock/meta.llama3-1-405b-instruct-v1:0",
    #     "bedrock/meta.llama3-1-70b-instruct-v1:0",
    #     "bedrock/meta.llama3-1-8b-instruct-v1:0",
    #     "bedrock/meta.llama3-70b-instruct-v1:0",
    #     "bedrock/meta.llama3-8b-instruct-v1:0",
    #     "bedrock/amazon.titan-text-lite-v1",
    #     "bedrock/amazon.titan-text-express-v1",
    #     "bedrock/cohere.command-text-v14",
    #     "bedrock/ai21.j2-mid-v1",
    #     "bedrock/ai21.j2-ultra-v1",
    #     "bedrock/ai21.jamba-instruct-v1:0",
    #     "bedrock/meta.llama2-13b-chat-v1",
    #     "bedrock/meta.llama2-70b-chat-v1",
    #     "bedrock/mistral.mistral-7b-instruct-v0:2",
    #     "bedrock/mistral.mixtral-8x7b-instruct-v0:1",
    # ],
}


PROVIDERS = [
    "openai",
    "anthropic",
    "gemini",
    "ollama",
    "watson",
    "bedrock",
    "azure",
    "cerebras",
    "llama",
]


"""
Max input token size by the model.
"""
LLM_CONTEXT_WINDOW_SIZES = {
    "gpt-3.5-turbo": 8192,
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

    "deepseek-chat": 128000,
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
    "claude-3-5-sonnet-2024102": 200000,
}


LLM_API_KEY_NAMES = {
    "openai":  "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

LLM_BASE_URL_KEY_NAMES = {
    "openai":  "OPENAI_API_BASE",
     "gemini": "GEMINI_API_BASE",
    "anthropic": "ANTHROPIC_API_BASE",
}

LLM_VARS = {
    "openai": [
        {
            "prompt": "Enter your OPENAI API key (press Enter to skip)",
            "key_name": "OPENAI_API_KEY",
        }
    ],
    "anthropic": [
        {
            "prompt": "Enter your ANTHROPIC API key (press Enter to skip)",
            "key_name": "ANTHROPIC_API_KEY",
        }
    ],
    "gemini": [
        {
            "prompt": "Enter your GEMINI API key (press Enter to skip)",
            "key_name": "GEMINI_API_KEY",
        }
    ],
    "watson": [
        {
            "prompt": "Enter your WATSONX URL (press Enter to skip)",
            "key_name": "WATSONX_URL",
        },
        {
            "prompt": "Enter your WATSONX API Key (press Enter to skip)",
            "key_name": "WATSONX_APIKEY",
        },
        {
            "prompt": "Enter your WATSONX Project Id (press Enter to skip)",
            "key_name": "WATSONX_PROJECT_ID",
        },
    ],
    "ollama": [
        {
            "default": True,
            "API_BASE": "http://localhost:11434",
        }
    ],
    "bedrock": [
        {
            "prompt": "Enter your AWS Access Key ID (press Enter to skip)",
            "key_name": "AWS_ACCESS_KEY_ID",
        },
        {
            "prompt": "Enter your AWS Secret Access Key (press Enter to skip)",
            "key_name": "AWS_SECRET_ACCESS_KEY",
        },
        {
            "prompt": "Enter your AWS Region Name (press Enter to skip)",
            "key_name": "AWS_REGION_NAME",
        },
    ],
    "azure": [
        {
            "prompt": "Enter your Azure deployment name (must start with 'azure/')",
            "key_name": "model",
        },
        {
            "prompt": "Enter your AZURE API key (press Enter to skip)",
            "key_name": "AZURE_API_KEY",
        },
        {
            "prompt": "Enter your AZURE API base URL (press Enter to skip)",
            "key_name": "AZURE_API_BASE",
        },
        {
            "prompt": "Enter your AZURE API version (press Enter to skip)",
            "key_name": "AZURE_API_VERSION",
        },
    ],
    "cerebras": [
        {
            "prompt": "Enter your Cerebras model name (must start with 'cerebras/')",
            "key_name": "model",
        },
        {
            "prompt": "Enter your Cerebras API version (press Enter to skip)",
            "key_name": "CEREBRAS_API_KEY",
        },
    ],
}



"""
Params for litellm.completion() func. Address common/unique params to each provider.
"""

PARAMS = {
    "litellm": [
        "api_base",
        "api_version,"
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
        "base_url",
        "api_key",
    ],
    "openai": [
        "timeout",
        # "temperature",
        # "top_p",
        # "n",
        # "stream",
        "stream_options",
        # "stop",
        "max_compl,etion_tokens",
        # "max_tokens",
        "modalities",
        "prediction",
        "audio",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        # "response_format",
        "seed",
        # "tools",
        # "tool_choice",
        "logprobs",
        "top_logprobs",
        "parallel_tool_calls",
        "extra_headers",
        "model_list"
    ],
    "gemini": [
        "topK",
    ]
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
