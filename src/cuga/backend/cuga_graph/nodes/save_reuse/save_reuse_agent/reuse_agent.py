from pathlib import Path
from typing import Any
import os

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from loguru import logger

from cuga.backend.activity_tracker.tracker import ActivityTracker
from cuga.backend.cuga_graph.nodes.shared.base_agent import BaseAgent
from cuga.backend.cuga_graph.state.agent_state import AgentState

from cuga.backend.cuga_graph.nodes.save_reuse.save_reuse_agent.utils.export_mcp import process_text_file
from cuga.backend.cuga_graph.nodes.save_reuse.save_reuse_agent.utils.save_reuse import consolidate_flow
from cuga.backend.llm.models import LLMManager
from cuga.backend.llm.utils.helpers import load_prompt_simple
from cuga.config import settings, PACKAGE_ROOT
import re

tracker = ActivityTracker()
llm_manager = LLMManager()


def ensure_parent_directory_exists(file_path):
    """Ensure the parent directory for the file exists."""
    file_path = Path(file_path)
    # Create parent directories if they don't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


class ReuseAgent(BaseAgent):
    def __init__(
        self,
        prompt_template: ChatPromptTemplate,
        llm: BaseChatModel,
        tools: Any = None,
        max_tokens: int = 15000,
    ):
        super().__init__()
        self.name = "ReuseAgent"

        # Override max_tokens for HTML generation (defaults to 15000 for this agent)
        # Try multiple attributes depending on LLM type
        if hasattr(llm, 'max_tokens'):
            llm.max_tokens = max_tokens
            logger.debug(f"ReuseAgent: Set llm.max_tokens to {max_tokens}")
        if hasattr(llm, 'max_completion_tokens'):
            llm.max_completion_tokens = max_tokens
            logger.debug(f"ReuseAgent: Set llm.max_completion_tokens to {max_tokens}")
        if hasattr(llm, 'model_kwargs') and llm.model_kwargs is not None:
            llm.model_kwargs['max_tokens'] = max_tokens
            logger.debug(f"ReuseAgent: Set llm.model_kwargs['max_tokens'] to {max_tokens}")

        self.chain = BaseAgent.get_chain(prompt_template=prompt_template, llm=llm, wx_json_mode="no_format")
        self.vischain = BaseAgent.get_chain(
            prompt_template=load_prompt_simple(
                "./prompts/explainbility.jinja2",
                "./prompts/explainbility_user.jinja2",
            ),
            llm=llm,
            wx_json_mode="no_format",
        )

    def output_parser(self, result: AIMessage, name) -> Any:
        result = AIMessage(content=result.content, name=name)
        return result

    def get_text_after_last_backticks(self, text):
        """Extract text after the last closing code fence, skipping the backticks themselves."""
        last_backticks_pos = text.rfind("```")
        if last_backticks_pos == -1:
            return ""  # No backticks found
        # Skip past the closing backticks and any newlines
        start_pos = last_backticks_pos + 3
        text_after = text[start_pos:].lstrip('\n')
        return text_after if text_after else ""

    def save_html_to_file(self, html_content, filename):
        """
        Save HTML content to a file.

        Args:
            html_content (str): The HTML content to save
            filename (str): The path/name of the file to save to
        """
        ensure_parent_directory_exists(filename)
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(html_content)
        logger.info(f"HTML content saved to {filename}")

    async def run(self, input_variables: AgentState, additional_utterance="") -> AIMessage:
        res = await consolidate_flow(self.chain, input_variables.input + ' ' + additional_utterance)

        # Handle case where no code was generated in the flow
        if res is None:
            logger.warning("ReuseAgent: No code found in flow trajectory")
            return AIMessage(
                content="⚠️ Cannot save this flow for reuse.\n\n"
                "Reason: This flow didn't involve any code generation steps (no CodeAgent actions).\n\n"
                "Only flows that generate Python code can be saved and reused."
            )

        pattern = r'```python\s*\n(.*?)\n```'
        matches = re.findall(pattern, res.content, re.DOTALL)
        if not matches:
            logger.warning("ReuseAgent: could not generate python")
            return AIMessage(
                content="❌ Failed to save flow for reuse.\n\nReason: Could not generate Python code from the flow.\n\n"
                + self.get_text_after_last_backticks(res.content)
            )

        # Generate HTML visualization only if enabled
        html_path = None
        if settings.advanced_features.save_reuse_generate_html:
            res_html = await self.vischain.ainvoke(input={"code": matches[0]})
            pattern = r'```html\s*\n(.*?)\n```'
            html_matches = re.findall(pattern, res_html.content, re.DOTALL)
            if not html_matches:
                logger.warning("ReuseAgent: could not generate html")
                return AIMessage(
                    content="⚠️ Partially saved flow for reuse.\n\n"
                    "✅ Python code was generated successfully\n\n"
                    "❌ Failed to generate HTML visualization\n\n"
                    + self.get_text_after_last_backticks(res.content)
                )
            # Save HTML visualization
            html_path = os.path.join(PACKAGE_ROOT, "backend", "server", "flows", "flow.html")
            self.save_html_to_file(html_matches[0], html_path)
        else:
            logger.info(
                "ReuseAgent: HTML generation disabled via settings.advanced_features.save_reuse_generate_html"
            )

        # Save Python code
        output_path = Path(
            os.path.join(PACKAGE_ROOT, "backend", "tools_env", "registry", "mcp_servers", "saved_flows.py")
        )
        ensure_parent_directory_exists(output_path)
        success = process_text_file(input_text=res.content, output_file=output_path)

        if not success:
            html_status = "✅ HTML visualization created\n\n" if html_path else ""
            return AIMessage(
                content="⚠️ Partially saved flow for reuse.\n\n"
                "✅ Python code was generated\n\n"
                f"{html_status}"
                f"❌ Failed to save Python code to: {output_path}\n\n"
                "Check the logs above for details on what went wrong.\n\n"
                + self.get_text_after_last_backticks(res.content)
            )

        html_message = (
            f"🎨 HTML visualization created: {html_path}\n\n"
            if html_path
            else "ℹ️ HTML visualization skipped (disabled in settings)\n\n"
        )
        return AIMessage(
            content="✅ Successfully saved flow for reuse!\n\n"
            "📝 Python code generated and saved\n\n"
            f"{html_message}"
            f"💾 Flow registered at: {output_path}\n\n"
            "You can now reuse this flow in future requests.\n\n"
            + self.get_text_after_last_backticks(res.content)
        )

    @staticmethod
    def create():
        dyna_model = settings.agent.planner.model
        pmt = load_prompt_simple(
            "./prompts/save_reuse.jinja2",
            "./prompts/save_reuse_user.jinja2",
        )
        # Use 15000 tokens for ReuseAgent to allow for long HTML generation
        return ReuseAgent(
            prompt_template=pmt,
            llm=llm_manager.get_model(dyna_model),
        )
