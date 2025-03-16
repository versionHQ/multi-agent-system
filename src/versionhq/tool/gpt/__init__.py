from dotenv import load_dotenv
load_dotenv(override=True)

import os
from openai import OpenAI
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
