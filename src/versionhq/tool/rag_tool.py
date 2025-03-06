import re
import requests
import gzip
import http.client
import urllib.request
from urllib.request import Request
from textwrap import dedent
from typing import Any, Optional, List, Dict

from pydantic import Field

from versionhq.agent.model import Agent
from versionhq.tool.model import BaseTool
from versionhq._utils.logger import Logger



class RagTool(BaseTool):
    """A Pydantic class to store a RAG tool object. Inherited from BaseTool"""

    api_key_name: str = Field(default=None)
    api_endpoint: Optional[str] = Field(default=None)

    url: Optional[str] = Field(default=None, description="url to scrape")
    headers: Optional[Dict[str, Any]] = Field(default_factory=dict, description="request headers")

    sources: Optional[List[Any]] = Field(default_factory=list, description="indexed data sources")
    query: Optional[str] = Field(default=None)
    text: Optional[str] = Field(default=None, description="text data source")


    def _sanitize_source_code(self, source_code: str | bytes = None) -> str | None:
        if not source_code:
            return None

        if isinstance(source_code, bytes):
            source_code = source_code.decode('utf-8')

        try:
            import html2text
        except:
            Logger().log(message="Dependencies for tools are missing. Add tool packages by running: `uv add versionhq[tool]`.", color="red", level="error")

        h = html2text.HTML2Text()
        h.ignore_links = False
        text = h.handle(source_code)
        text = re.sub(r"[^a-zA-Z$0-9\s\n]", "", text)
        return dedent(text)


    def _scrape_url(self, url: str = None) -> str | None:
        url = url if url else self.url

        if not url:
            return None

        http.client.HTTPConnection.debuglevel = 1

        try:
            req = Request(url=url, headers=self.headers, origin_req_host=url, method="GET")
            res = ""

            with urllib.request.urlopen(req) as url:
                if url.info().get("Content-Encoding") == "gzip":
                    res = gzip.decompress(url.read())
                else:
                    res = url.read()

            text = self._sanitize_source_code(source_code=res)
            return text

        except requests.exceptions.HTTPError as e:
            Logger().log(level="error", message=f"HTTP error occurred: {str(e)}", color="red")
            return None

        except Exception as e:
            Logger().log(level="error", message=f"Error fetching URL {self.api_endpoint}: {str(e)}", color="red")
            return None


    def store_data(self, agent: Agent = None) -> None:
        """Stores retrieved data in the storage"""
        if not agent:
            return

        text = self.text if self.text else self._scrape_url(self.url)
        self.text = text
        knowledge_sources = [*agent.knowledge_sources, str(text), ] if agent.knowledge_sources else [str(text),]
        agent.update(knowledge_sources=knowledge_sources)


    def _run(self, agent: Agent = None, query: str = None) -> List[str]:
        query = query if query else self.query

        if not query or not agent:
            text = self.text if self.text else self._scrape_url(self.url)
            self.text = text
            return [text,]

        else:
            results, res = [], []
            if agent._knowledge:
                res = agent._knowledge.query(query=[query], limit=5)
            else:
                self.store_data(agent=agent)
                res = agent._knowledge.query(query=[query], limit=5)

            for item in res:
                if isinstance(item, dict):
                    results.append(item["context"])
                else:
                    results.append(str(item))
            return results


    def run(self, *args, **kwargs):
        return self._run(*args, **kwargs)
