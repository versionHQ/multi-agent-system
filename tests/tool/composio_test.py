import os

from versionhq.tool.composio_tool import ComposioHandler
from versionhq.tool.composio_tool_vars import ComposioAppName, ComposioAuthScheme, ComposioStatus, ComposioAction

DEFAULT_MODEL_NAME = os.environ.get("DEFAULT_MODEL_NAME", "gpt-3.5-turbo")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY")
TEST_CONNECTED_ACCOUNT_ID = os.environ.get("TEST_CONNECTED_ACCOUNT_ID", "f79b1a07-db14-4aa5-959e-5713ccd18de2")


def _test_connect_with_composio():
    composio = ComposioHandler(
        app_name=ComposioAppName.HUBSPOT,
        auth_scheme=ComposioAuthScheme.OAUTH2,
        connected_account_id=TEST_CONNECTED_ACCOUNT_ID
    )
    composio, status = composio._connect()
    assert composio.connected_account_id is not None
    assert status is not ComposioStatus.FAILED


def _test_execute_action_on_langchain():
    composio = ComposioHandler(
        app_name=ComposioAppName.HUBSPOT,
        auth_scheme=ComposioAuthScheme.OAUTH2,
        connected_account_id=TEST_CONNECTED_ACCOUNT_ID
    )
    composio, status = composio._connect()
    composio, res = composio.execute_composio_action_with_langchain(
        action_name=ComposioAction.HUBSPOT_CREATE_PIPELINE_STAGE,
        task_in_natural_language="create a demo pipeline"
    )
    assert status == ComposioStatus.ACTIVE.value
    assert composio.connected_account_id == TEST_CONNECTED_ACCOUNT_ID
    assert composio.tools is not None
    assert res is not None
    assert isinstance(res, str)
