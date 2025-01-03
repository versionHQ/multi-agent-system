import os

from versionhq.tool.composio import Composio
from versionhq.tool import ComposioAppName, ComposioAuthScheme, COMPOSIO_STATUS

DEFAULT_MODEL_NAME = os.environ.get("LITELLM_MODEL_NAME", "gpt-3.5-turbo")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY")


def test_connect_composio():
    composio = Composio(app_name=ComposioAppName.HUBSPOT, auth_scheme=ComposioAuthScheme.OAUTH2)
    account_id, status = composio.connect()

    assert account_id is not None
    assert status is not COMPOSIO_STATUS.FAILED


if __name__ == "__main__":
    test_connect_composio()
