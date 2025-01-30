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


    def build_context_for_task(self, task, context: List[Any] | str) -> str:
        """
        Automatically builds a minimal, highly relevant set of contextual information for a given task.
        """

        query = f"{task.description} {context}".strip()

        if query == "":
            return ""

        context = []
        context.append(self._fetch_stm_context(query))
        if self.memory_provider == "mem0":
            context.append(self._fetch_user_context(query))
        return "\n".join(filter(None, context))


    def _fetch_stm_context(self, query) -> str:
        """
        Fetches recent relevant insights from STM related to the task's description and expected_output, formatted as bullet points.
        """
        stm_results = self.stm.search(query)
        formatted_results = "\n".join(
            [
                f"- {result['memory'] if self.memory_provider == 'mem0' else result['context']}"
                for result in stm_results
            ]
        )
        return f"Recent Insights:\n{formatted_results}" if stm_results else ""


    def _fetch_ltm_context(self, task) -> Optional[str]:
        """
        Fetches historical data or insights from LTM that are relevant to the task's description and expected_output, formatted as bullet points.
        """
        ltm_results = self.ltm.search(task, latest_n=2)
        if not ltm_results:
            return None

        formatted_results = [suggestion for result in ltm_results for suggestion in result["metadata"]["suggestions"]]
        formatted_results = list(dict.fromkeys(formatted_results))
        formatted_results = "\n".join([f"- {result}" for result in formatted_results])
        return f"Historical Data:\n{formatted_results}" if ltm_results else ""


    def _fetch_user_context(self, query: str) -> str:
        """
        Fetches and formats relevant user information from User Memory.
        """

        user_memories = self.um.search(query)
        if not user_memories:
            return ""

        formatted_memories = "\n".join(f"- {result['memory']}" for result in user_memories)
        return f"User memories/preferences:\n{formatted_memories}"


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
