from typing import Any, Dict, Optional, List

from versionhq.memory.model import ShortTermMemory, LongTermMemory, UserMemory


class ContextualMemory:
    """
    A class to construct context from memories (ShortTermMemory, UserMemory).
    The context will be added to the prompt when the agent executes the task.
    """

    def __init__(
        self,
        memory_config: Optional[Dict[str, Any]],
        stm: ShortTermMemory,
        ltm: LongTermMemory,
        um: UserMemory,
        # em: EntityMemory,
    ):
        self.memory_provider = memory_config.get("provider") if memory_config is not None else None
        self.stm = stm
        self.ltm = ltm
        self.um = um


    def _sanitize_query(self, query: str = None) -> str:
        if not query:
            return ""

        return query.replace("{", "").replace("}", "")


    def _fetch_stm_context(self, query: str = None) -> str:
        """
        Fetches recent relevant insights from STM
        """
        if not query:
            return ""


        stm_results = self.stm.search(query)
        if not stm_results:
            return ""

        formatted_results = "\n".join(
            [
                f"- {result['memory'] if self.memory_provider == 'mem0' else result['context']}"
                for result in stm_results
            ]
        )
        return f"Recent Insights:\n{formatted_results}" if stm_results else ""


    def _fetch_ltm_context(self, query: str = None) -> Optional[str]:
        """
        Fetches historical data or insights from LTM that are relevant to the task's description and expected_output, formatted as bullet points.
        """
        if not query:
            return ""

        ltm_results = self.ltm.search(query, latest_n=2)
        if not ltm_results:
            return ""

        formatted_results = [suggestion for result in ltm_results for suggestion in result["metadata"]["suggestions"]]
        formatted_results = list(dict.fromkeys(formatted_results))
        formatted_results = "\n".join([f"- {result}" for result in formatted_results])
        return f"Historical Data:\n{formatted_results}" if ltm_results else ""


    def _fetch_user_context(self, query: str = None) -> str:
        """
        Fetches and formats relevant user information from User Memory.
        """
        if not query:
            return ""

        user_memories = self.um.search(query)
        if not user_memories:
            return ""

        formatted_memories = "\n".join(f"- {result['memory']}" for result in user_memories)
        return f"User memories/preferences:\n{formatted_memories}"


    def build_context_for_task(self, query: str = None) -> str:
        """
        Automatically builds a minimal, highly relevant set of contextual information for a given task.
        """
        if not query:
            return ""

        query = self._sanitize_query(query=query)

        context = []
        context.append(self._fetch_stm_context(query))
        context.append(self._fetch_ltm_context(query))
        if self.memory_provider == "mem0":
            context.append(self._fetch_user_context(query))
        return "\n".join(filter(None, context))


    # def _fetch_entity_context(self, query) -> str:
    #     """
    #     Fetches relevant entity information from Entity Memory related to the task's description and expected_output,
    #     formatted as bullet points.
    #     """
    #     em_results = self.em.search(query)
    #     formatted_results = "\n".join(
    #         [
    #             f"- {result['memory'] if self.memory_provider == 'mem0' else result['context']}"
    #             for result in em_results
    #         ]  # type: ignore #  Invalid index type "str" for "str"; expected type "SupportsIndex | slice"
    #     )
    #     return f"Entities:\n{formatted_results}" if em_results else ""
