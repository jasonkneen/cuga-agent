from typing import Any, Dict, List, Optional

from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import CugaAgent as BaseCugaAgent
from cuga.backend.cuga_graph.nodes.cuga_lite.combined_tool_provider import CombinedToolProvider


class CugaAgent(BaseCugaAgent):
    """Fast CUGA agent that automatically loads tools from the registry.

    This is a convenience wrapper around BaseCugaAgent that uses ToolRegistryProvider.
    It maintains the same interface as the original fast.py CugaAgent.

    Usage:
        agent = CugaAgent()
        await agent.initialize()
        answer, metrics, state_messages, chat_messages = await agent.execute("Your task here")

    With custom instructions:
        agent = CugaAgent(instructions="Always be helpful and concise.")
        await agent.initialize()
        answer, metrics, state_messages, chat_messages = await agent.execute("Your task here")

    With Langfuse tracing:
        try:
            from langfuse.langchain import CallbackHandler
        except ImportError:
            from langfuse.callback.langchain import LangchainCallbackHandler as CallbackHandler
        langfuse_handler = CallbackHandler() if settings.advanced_features.langfuse_tracing else None
        agent = CugaAgent(langfuse_handler=langfuse_handler)
        await agent.initialize()
        answer, metrics, state_messages, chat_messages = await agent.execute("Your task here")
    """

    def __init__(
        self,
        app_names: Optional[List[str]] = None,
        model_settings: Optional[Dict] = None,
        langfuse_handler: Optional[Any] = None,
        instructions: Optional[str] = None,
        task_loaded_from_file: bool = False,
    ):
        """Initialize CugaAgent.

        Args:
            app_names: Optional list of specific app names to load. If None, loads all available apps.
            model_settings: Optional model settings to override defaults.
            langfuse_handler: Optional Langfuse callback handler for tracing.
            instructions: Optional custom instructions to provide to the agent.
            task_loaded_from_file: If True, indicates that the task was loaded from a file.
        """
        tool_provider = CombinedToolProvider(app_names=app_names)
        super().__init__(
            tool_provider=tool_provider,
            model_settings=model_settings,
            langfuse_handler=langfuse_handler,
            instructions=instructions,
            task_loaded_from_file=task_loaded_from_file,
        )
        self.app_names = app_names


__all__ = ['CugaAgent']
