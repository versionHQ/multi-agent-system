from enum import Enum

DEFAULT_AUTH_SCHEME = "OAUTH2"

class ComposioAuthScheme(str, Enum):
    OAUTH2 = "OAUTH2"
    BEARER_TOKEN = "BEARER_TOKEN"
    API_KEY = "API_KEY"


class ComposioAppName(str, Enum):
    """
    Enum to store app names that we can connect via Composio as data pipelines or destination services.
    """

    SALESFORCE = "salesforce"
    AIRTABLE = "airtable"
    MAILCHIMP = "mailchimp"
    HUBSPOT = "hubspot"
    KLAVIYO = "klaviyo"
    GOOGLESHEET = "googlesheets"
    GMAIL = "gmail"
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    TWITTER_MEDIA = "twitter_media"
    LINKEDIN = "linkedin"


composio_app_set = [
    (ComposioAppName.SALESFORCE, ComposioAuthScheme.OAUTH2),
    (ComposioAppName.AIRTABLE, ComposioAuthScheme.OAUTH2, ComposioAuthScheme.API_KEY, ComposioAuthScheme.BEARER_TOKEN),
    (ComposioAppName.MAILCHIMP, ComposioAuthScheme.OAUTH2),
    (ComposioAppName.HUBSPOT, ComposioAuthScheme.OAUTH2, ComposioAuthScheme.BEARER_TOKEN),
    (ComposioAppName.KLAVIYO, ComposioAuthScheme.OAUTH2, ComposioAuthScheme.API_KEY),
    (ComposioAppName.GOOGLESHEET, ComposioAuthScheme.OAUTH2),
    (ComposioAppName.GMAIL, ComposioAuthScheme.OAUTH2, ComposioAuthScheme.BEARER_TOKEN),
    (ComposioAppName.TWITTER, ComposioAuthScheme.OAUTH2),
    (ComposioAppName.TWITTER_MEDIA, ComposioAuthScheme.OAUTH2),
    (ComposioAppName.FACEBOOK, ComposioAuthScheme.OAUTH2),
    (ComposioAppName.LINKEDIN, ComposioAuthScheme.OAUTH2),
]

class ComposioStatus(str, Enum):
    INITIATED = "INITIATED"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"




class ComposioAction(str, Enum):
    """
    Enum to store composio's action that can be called via `Actions.xxx`
    """
    # HUBSPOT_INITIATE_DATA_IMPORT_PROCESS = "hubspot_initate_date_import_process"
    HUBSPOT_CREATE_PIPELINE_STAGE = "hubspot_create_pipeline_stage"
