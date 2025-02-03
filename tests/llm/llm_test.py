import os
import pytest
import litellm
from unittest.mock import patch

from versionhq.llm.llm_vars import MODELS, LLM_CONTEXT_WINDOW_SIZES
from versionhq.llm.model import LLM, DEFAULT_CONTEXT_WINDOW_SIZE, DEFAULT_MODEL_NAME


def dummy_func() -> str:
    return "dummy"


def test_create_llm_from_valid_name():
    """
    Make sure base params are set properly in the LLM class of all the models available in the framework.
    """

    for k, v in MODELS.items():
        for model_name in v:
            llm = LLM(model=model_name, callbacks=[dummy_func,])

            assert llm._init_model_name == model_name
            assert llm.model == model_name
            assert llm.provider == k
            assert llm.context_window_size == int(LLM_CONTEXT_WINDOW_SIZES.get(model_name) *0.75) if LLM_CONTEXT_WINDOW_SIZES.get(model_name) is not None else DEFAULT_CONTEXT_WINDOW_SIZE
            assert llm._supports_function_calling() is not None
            assert llm._supports_stop_words() is not None
            assert litellm.callbacks == [dummy_func,]


def test_create_llm_from_invalid_name():
    """
    Test if all the params will be set properly with a givne invalid model name.
    """
    llm = LLM(model="4o", callbacks=[dummy_func,])

    assert llm._init_model_name == "4o"
    assert llm.model == "gpt-4o"
    assert llm.provider == "openai"
    assert llm.context_window_size == int(128000 * 0.75)
    assert llm._supports_function_calling() == True
    assert llm._supports_stop_words() == True
    assert litellm.callbacks == [dummy_func,]


def test_create_llm_from_provider():
    llm = LLM(provider="gemini", callbacks=[dummy_func,])

    assert llm.model == "gemini/gemini-1.5-flash"
    assert llm.provider == "gemini"
    assert llm.context_window_size == int(LLM_CONTEXT_WINDOW_SIZES.get(llm.model) *0.75)
    assert llm._supports_function_calling() == True
    assert llm._supports_stop_words() == True
    assert litellm.callbacks == [dummy_func,]
