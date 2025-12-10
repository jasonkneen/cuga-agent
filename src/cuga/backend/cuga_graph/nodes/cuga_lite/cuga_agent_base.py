"""
CugaAgent Base Class

Core CUGA agent that works with different tool providers through a unified interface.
"""

import asyncio
import contextlib
import io
import time
import types
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.callbacks.usage import UsageMetadataCallbackHandler
from langchain_core.tools import StructuredTool
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

try:
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
except ImportError:
    try:
        from langfuse.callback.langchain import LangchainCallbackHandler as LangfuseCallbackHandler
    except ImportError:
        LangfuseCallbackHandler = None

from cuga.backend.cuga_graph.nodes.api.code_agent.code_act_agent import create_codeact
from cuga.backend.cuga_graph.state.agent_state import VariablesManager
from cuga.backend.llm.models import LLMManager
from cuga.backend.cuga_graph.nodes.cuga_lite.tool_provider_interface import (
    ToolProviderInterface,
    AppDefinition,
)
from cuga.config import settings


class CombinedMetricsCallback(BaseCallbackHandler):
    """Combined callback handler that tracks both timing and token usage."""

    def __init__(self):
        super().__init__()
        self.start_time = None
        self.end_time = None
        self.llm_calls = 0
        self.usage_callback = UsageMetadataCallbackHandler()

    def reset(self):
        """Reset all metrics."""
        self.start_time = time.time()
        self.end_time = None
        self.llm_calls = 0
        self.usage_callback.usage_metadata = {}

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called when LLM starts."""
        if self.start_time is None:
            self.start_time = time.time()
        self.llm_calls += 1

    def on_llm_end(self, response, **kwargs):
        """Called when LLM ends."""
        self.end_time = time.time()
        self.usage_callback.on_llm_end(response, **kwargs)

    def get_total_tokens(self):
        """Get total tokens across all models."""
        total = 0
        for model_usage in self.usage_callback.usage_metadata.values():
            if isinstance(model_usage, dict):
                if 'total_tokens' in model_usage:
                    total += model_usage['total_tokens']
                elif 'input_tokens' in model_usage and 'output_tokens' in model_usage:
                    total += model_usage['input_tokens'] + model_usage['output_tokens']
            elif hasattr(model_usage, 'total_tokens'):
                total += model_usage.total_tokens
            elif hasattr(model_usage, 'input_tokens') and hasattr(model_usage, 'output_tokens'):
                total += model_usage.input_tokens + model_usage.output_tokens
        return total

    def get_metrics(self):
        """Get current metrics."""
        duration = (self.end_time or time.time()) - (self.start_time or time.time())
        return {
            'duration_seconds': round(duration, 2),
            'llm_calls': self.llm_calls,
            'total_tokens': self.get_total_tokens(),
            'usage_by_model': self.usage_callback.usage_metadata,
        }

    def print_summary(self):
        """Print a summary of metrics."""
        metrics = self.get_metrics()
        print("\n📊 Execution Metrics:")
        print(f"   Duration: {metrics['duration_seconds']}s")
        print(f"   LLM Calls: {metrics['llm_calls']}")
        print(f"   Total Tokens: {metrics['total_tokens']}")
        if metrics['usage_by_model']:
            print(f"   Usage by model: {metrics['usage_by_model']}")
        return metrics


class CugaAgent:
    """
    Base CUGA agent that works with different tool providers.

    This agent supports multiple tool interfaces:
    - ToolRegistryProvider: Tools from MCP registry (separate process)
    - DirectLangChainToolsProvider: LangChain tools passed directly (in-process)

    Usage with Registry:
        ```python
        from cuga.backend.cuga_graph.nodes.cuga_lite.tool_registry_provider import ToolRegistryProvider

        provider = ToolRegistryProvider(app_names=["digital_sales"])
        agent = CugaAgent(tool_provider=provider)
        await agent.initialize()
        answer, metrics, state_messages, chat_messages = await agent.execute("Get top accounts")
        ```

    Usage with Direct Tools:
        ```python
        from cuga.backend.cuga_graph.nodes.cuga_lite.direct_langchain_tools_provider import DirectLangChainToolsProvider
        from langchain_core.tools import tool

        @tool
        def my_tool(query: str) -> str:
            '''Custom tool'''
            return "result"

        provider = DirectLangChainToolsProvider(tools=[my_tool])
        agent = CugaAgent(tool_provider=provider)
        await agent.initialize()
        answer, metrics, state_messages, chat_messages = await agent.execute("Use my tool")
        ```

    Usage with Custom Return Cases:
        ```python
        provider = ToolRegistryProvider(app_names=["digital_sales"])
        custom_cases = [
            "You have a complete final answer with all necessary data from code execution",
            "You need missing parameters or clarification from the user",
            "You need user approval before executing a destructive action",
            "You encounter an ambiguous situation that requires user decision"
        ]
        agent = CugaAgent(
            tool_provider=provider,
            allow_user_clarification=True,
            override_return_to_user_cases=custom_cases
        )
        await agent.initialize()
        answer, metrics, state_messages, chat_messages = await agent.execute("Delete account 123")
        ```
    """

    @staticmethod
    def create_langfuse_handler():
        """Create a Langfuse callback handler if tracing is enabled in settings."""
        if settings.advanced_features.langfuse_tracing:
            if LangfuseCallbackHandler is not None:
                return LangfuseCallbackHandler()
            else:
                logger.warning("Langfuse tracing enabled but langfuse package not available")
        return None

    def __init__(
        self,
        tool_provider: ToolProviderInterface,
        model_settings: Optional[Dict] = None,
        langfuse_handler: Optional[Any] = None,
        eval_fn: Optional[Any] = None,
        prompt_template: Optional[str] = None,
        allow_user_clarification: bool = True,
        override_return_to_user_cases: Optional[List[str]] = None,
        instructions: Optional[str] = None,
        task_loaded_from_file: bool = False,
    ):
        """
        Initialize CugaAgent.

        Args:
            tool_provider: Tool provider implementation (ToolRegistryProvider or DirectLangChainToolsProvider)
            model_settings: Optional model settings to override defaults
            langfuse_handler: Optional Langfuse callback handler for tracing
            eval_fn: Optional custom evaluation function for code execution
            prompt_template: Optional custom prompt template
            allow_user_clarification: If True, agent can ask user for clarification. If False, only final answers allowed.
            override_return_to_user_cases: Optional list of custom cases (in natural language) when agent should return to user.
                                  If None, uses default cases. Example: ["Request user approval for destructive actions"]
            instructions: Optional special instructions to include in the system prompt.
            task_loaded_from_file: If True, indicates that the task was loaded from a file (e.g., markdown file).
        """
        self.tool_provider = tool_provider
        self.model_settings = model_settings
        self.langfuse_handler = langfuse_handler
        self.eval_fn = eval_fn
        self.prompt_template = prompt_template
        self.allow_user_clarification = allow_user_clarification
        self.override_return_to_user_cases = override_return_to_user_cases
        self.instructions = instructions
        self.task_loaded_from_file = task_loaded_from_file

        self.apps: List[AppDefinition] = []
        self.tools: List[StructuredTool] = []
        self.agent = None
        self.initialized = False

    def get_langfuse_trace_id(self) -> Optional[str]:
        """Get the current Langfuse trace ID if available."""
        if self.langfuse_handler and hasattr(self.langfuse_handler, 'last_trace_id'):
            return self.langfuse_handler.last_trace_id
        return None

    async def initialize(self):
        """Initialize the agent by loading tools from the provider."""
        logger.info("Initializing CugaAgent...")

        await self.tool_provider.initialize()

        self.apps = await self.tool_provider.get_apps()
        logger.info(f"Found {len(self.apps)} apps: {[app.name for app in self.apps]}")

        self.tools = await self.tool_provider.get_all_tools()
        if not self.tools:
            raise Exception("No tools available from tool provider")

        logger.info(f"Successfully loaded {len(self.tools)} tools")

        llm_manager = LLMManager()
        if self.model_settings:
            model_config = self.model_settings
        else:
            model_config = settings.agent.code.model.copy()
            model_config["streaming"] = False

        model = llm_manager.get_model(model_config)
        logger.info(f"Initialized LLM: {type(model).__name__}")

        if self.prompt_template:
            custom_prompt = self.prompt_template
        else:
            custom_prompt = create_mcp_prompt(
                self.tools,
                allow_user_clarification=self.allow_user_clarification,
                return_to_user_cases=self.override_return_to_user_cases,
                instructions=self.instructions,
                apps=self.apps,
                task_loaded_from_file=self.task_loaded_from_file,
            )

        for tool in self.tools:
            if not hasattr(tool, 'func'):
                logger.warning(f"Tool {tool.name} missing .func attribute, attempting to add it")
                if hasattr(tool, 'coroutine') and tool.coroutine:
                    tool.func = tool.coroutine
                elif hasattr(tool, '_run'):
                    tool.func = tool._run

        if self.eval_fn:
            eval_function = self.eval_fn
        else:
            eval_function = eval_with_tools_async

        agent_graph = create_codeact(
            model=model, tools=self.tools, eval_fn=eval_function, prompt=custom_prompt
        )

        self.agent = agent_graph.compile()
        self.initialized = True
        logger.info("CugaAgent initialized successfully")

    async def execute(
        self,
        task: str,
        recursion_limit: int = 15,
        show_progress: bool = True,
        state_messages: Optional[List] = None,
        chat_messages: Optional[List[BaseMessage]] = None,
        initial_context: Optional[Dict[str, Any]] = None,
        keep_last_n_vars: int = 4,
    ) -> Tuple[str, Dict[str, Any], Optional[List], Optional[List[BaseMessage]]]:
        """
        Execute a task using the CodeAct agent.

        Args:
            task: The task description to execute
            recursion_limit: Maximum number of reasoning steps
            show_progress: Whether to print progress messages
            state_messages: Optional list to append messages to (for graph visualization)
            chat_messages: Optional chat history to include in context
            initial_context: Optional initial context/variables for CodeAct state
            keep_last_n_vars: Number of most recent variables to keep in context (default: 2)

        Returns:
            Tuple of (answer, usage_metrics, state_messages, updated_chat_messages)
        """
        if not self.initialized:
            raise Exception("Agent not initialized. Call await agent.initialize() first")

        if show_progress:
            print(f"\n{'=' * 60}")
            print(f"🚀 CugaAgent executing: {task}")
            print(f"{'=' * 60}")

        callbacks = []
        metrics_callback = CombinedMetricsCallback()
        callbacks.append(metrics_callback)

        agent = self.agent

        if self.langfuse_handler and settings.advanced_features.langfuse_tracing:
            callbacks.append(self.langfuse_handler)
            if show_progress:
                print("🔍 Langfuse tracing enabled")

        config = {"thread_id": 1, "recursion_limit": recursion_limit, "callbacks": callbacks}

        initial_messages = []
        if chat_messages:
            logger.debug(f"Chat messages: {chat_messages}")
            for msg in chat_messages:
                if isinstance(msg, HumanMessage):
                    initial_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    initial_messages.append({"role": "assistant", "content": msg.content})
                elif hasattr(msg, 'content'):
                    role = getattr(msg, 'type', 'user')
                    if role == 'human' or role == 'user':
                        initial_messages.append({"role": "user", "content": msg.content})
                    elif role == 'ai' or role == 'assistant':
                        initial_messages.append({"role": "assistant", "content": msg.content})

        # Prepare task content with variables summary if needed
        task_content = task

        if initial_context and not chat_messages:
            # If we have initial context but no chat history, present the variables
            from cuga.backend.cuga_graph.state.agent_state import VariablesManager

            var_manager = VariablesManager()
            variable_names = list(initial_context.keys())
            if variable_names:
                variables_summary = var_manager.get_variables_summary(variable_names=variable_names)
                task_content = f"{task}\n\n## Available Variables\n\n{variables_summary}"
                logger.info(
                    f"Added variables summary for {len(variable_names)} variables to first user message"
                )

        initial_messages.append({"role": "user", "content": task_content})

        initial_state = {"messages": initial_messages, "context": initial_context or {}}

        if show_progress:
            print("🤖 Starting CodeAct Agent Execution")
            print("=" * 60)

        execution_steps = []
        all_code = []
        all_execution_outputs = []

        def extract_code_from_content(content: str) -> str:
            """Extract code from markdown code blocks in message content."""
            import re

            BACKTICK_PATTERN = r"```(.*?)(?:```(?:\n|$))"
            code_blocks = re.findall(BACKTICK_PATTERN, content, re.DOTALL)
            if not code_blocks:
                return ""

            processed_blocks = []
            for block in code_blocks:
                block = block.strip()
                lines = block.split("\n")
                if lines and (not lines[0].strip() or " " not in lines[0].strip()):
                    block = "\n".join(lines[1:])
                processed_blocks.append(block)
            return "\n\n".join(processed_blocks)

        try:
            step_count = 0
            final_state = None
            last_code = None

            async for chunk in agent.astream(initial_state, config=config, stream_mode="values"):
                step_count += 1
                final_state = chunk

                # logger.debug(f"Chunk keys: {list(chunk.keys())}")
                logger.debug(f"Has script: {'script' in chunk}")

                if "script" in chunk and chunk["script"]:
                    code = chunk["script"]
                    if code and code not in all_code:
                        all_code.append(code)
                        last_code = code
                        execution_steps.append(f"Step {step_count}: Code generation")
                        logger.debug(f"Captured code from script field (length: {len(code)})")

                if "messages" in chunk and chunk["messages"]:
                    last_msg = chunk["messages"][-1]
                    if hasattr(last_msg, "content"):
                        content = last_msg.content

                        if isinstance(content, str) and "Execution output:" in content:
                            execution_output = content.replace("Execution output:\n", "")
                            all_execution_outputs.append(execution_output)

                            if state_messages is not None:
                                import json

                                state_messages.append(
                                    AIMessage(
                                        content=json.dumps(
                                            {
                                                "status": "execution_output",
                                                "step": step_count,
                                                "code": last_code or "",
                                                "execution_output": execution_output,
                                                "message": f"Code execution and output for step {step_count}",
                                            }
                                        )
                                    )
                                )

                        if show_progress:
                            role = "AI" if "AIMessage" in str(last_msg.__class__) else "User"
                            display_content = content
                            if len(display_content) > 5000:
                                display_content = display_content[:5000] + "..."
                            print(f"\n[{role}]: {display_content}")

            if show_progress:
                print(f"\n{'=' * 60}")
                print(f"✅ Execution completed in {step_count} steps")
                print(f"{'=' * 60}")

            final_answer = "No answer found"
            if final_state and "messages" in final_state:
                # Find the last AI message with non-empty content
                for msg in reversed(final_state["messages"]):
                    logger.debug(f"Message: {msg}")
                    if hasattr(msg, "__class__") and "AIMessage" in str(msg.__class__):
                        content = msg.content if hasattr(msg, 'content') else str(msg)
                        logger.debug(f"Content: {content}")
                        # Only use non-empty content as final answer
                        if content and content.strip():
                            final_answer = content
                            break

                # If no text answer found, use the last execution output as the answer
                if final_answer == "No answer found" and all_execution_outputs:
                    # Extract the actual output (before the "New Variables" section)
                    last_output = all_execution_outputs[-1]
                    if "## New Variables Created:" in last_output:
                        actual_output = last_output.split("## New Variables Created:")[0].strip()
                    else:
                        actual_output = last_output.strip()

                    if actual_output:
                        final_answer = actual_output
                        logger.info(
                            "Using last execution output as final answer (no text answer provided by agent)"
                        )

            if show_progress:
                print(f"\n{'=' * 60}")
                print("FINAL ANSWER:")
                print(f"{'=' * 60}")
                print(final_answer)

            usage_metrics = metrics_callback.get_metrics()
            usage_metrics['step_count'] = step_count
            usage_metrics['tools_available'] = len(self.tools)
            usage_metrics['apps_used'] = [app.name for app in self.apps]

            if state_messages is not None:
                from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_lite_node import CugaLiteOutput

                combined_code = "\n\n".join(all_code) if all_code else ""
                combined_execution_output = (
                    "\n\n".join(all_execution_outputs) if all_execution_outputs else ""
                )

                output = CugaLiteOutput(
                    code=combined_code,
                    execution_output=combined_execution_output,
                    steps_summary=execution_steps,
                    summary=f"Task completed successfully in {step_count} steps",
                    metrics=usage_metrics,
                    final_answer=final_answer,
                )

                final_context = {}
                if final_state and "context" in final_state:
                    full_context = final_state["context"]

                    # Separate initial context variables from newly created ones
                    initial_var_names = set(initial_context.keys()) if initial_context else set()
                    new_var_names = [k for k in full_context.keys() if k not in initial_var_names]

                    # Keep initial context variables (convert sets to lists for serialization)
                    final_context = {}
                    for k, v in full_context.items():
                        if k in initial_var_names:
                            if isinstance(v, set):
                                final_context[k] = list(v)
                                logger.debug(f"Converted set variable '{k}' to list in final_context")
                            else:
                                final_context[k] = v

                    # Apply keep_last_n_vars only to newly created variables
                    if keep_last_n_vars > 0 and len(new_var_names) > keep_last_n_vars:
                        # Keep only the last N newly created variables
                        vars_to_keep = new_var_names[-keep_last_n_vars:]
                        for var_name in vars_to_keep:
                            value = full_context[var_name]
                            if isinstance(value, set):
                                final_context[var_name] = list(value)
                                logger.debug(f"Converted set variable '{var_name}' to list in final_context")
                            else:
                                final_context[var_name] = value

                        # Remove newly created variables that are not being kept from var_manager
                        from cuga.backend.cuga_graph.state.agent_state import VariablesManager

                        var_manager = VariablesManager()
                        vars_to_remove = new_var_names[:-keep_last_n_vars]  # All except last N new vars
                        for var_name in vars_to_remove:
                            if var_manager.remove_variable(var_name):
                                logger.debug(f"Removed variable '{var_name}' from var_manager")

                        logger.debug(
                            f"Kept last {keep_last_n_vars} of {len(new_var_names)} newly created variables (plus {len(initial_var_names)} initial vars)"
                        )
                    else:
                        # Keep all newly created variables
                        for var_name in new_var_names:
                            value = full_context[var_name]
                            if isinstance(value, set):
                                final_context[var_name] = list(value)
                                logger.debug(f"Converted set variable '{var_name}' to list in final_context")
                            else:
                                final_context[var_name] = value
                        logger.debug(
                            f"Preserving all {len(new_var_names)} newly created variables (plus {len(initial_var_names)} initial vars)"
                        )

                state_messages.append(
                    AIMessage(content=output.model_dump_json(), additional_kwargs={"context": final_context})
                )

            if show_progress:
                print("\n📊 Execution Metrics:")
                print(f"   Duration: {usage_metrics['duration_seconds']}s")
                print(f"   LLM Calls: {usage_metrics['llm_calls']}")
                print(f"   Total Tokens: {usage_metrics['total_tokens']}")
                print(f"   Steps: {step_count}")
                print(f"   Tools Available: {len(self.tools)}")

                trace_id = self.get_langfuse_trace_id()
                if trace_id:
                    print(f"   Langfuse Trace ID: {trace_id}")
                    logger.info(f"Langfuse Trace ID: {trace_id}")

            if state_messages is None:
                state_messages = []

            # Convert final_state["messages"] back to chat_messages format for conversation history
            updated_chat_messages = None
            if final_state and "messages" in final_state and chat_messages is not None:
                updated_chat_messages = []
                for msg in final_state["messages"]:
                    if isinstance(msg, dict):
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        if role == "user":
                            updated_chat_messages.append(HumanMessage(content=content))
                        elif role == "assistant":
                            updated_chat_messages.append(AIMessage(content=content))
                    elif isinstance(msg, (HumanMessage, AIMessage)):
                        updated_chat_messages.append(msg)
                logger.info(f"Converted {len(updated_chat_messages)} messages for chat history continuation")

            return final_answer, usage_metrics, state_messages, updated_chat_messages

        except asyncio.TimeoutError:
            logger.error("Execution timeout")
            usage_metrics = metrics_callback.get_metrics()
            usage_metrics['error'] = 'timeout'
            return "Error: Execution timeout", usage_metrics, state_messages, None
        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
            usage_metrics = metrics_callback.get_metrics()
            usage_metrics['error'] = 'interrupted'
            return "Error: Interrupted by user", usage_metrics, state_messages, None
        except Exception as e:
            logger.error(f"Error during execution: {e}", exc_info=True)
            usage_metrics = metrics_callback.get_metrics()
            usage_metrics['error'] = str(e)
            return f"Error during execution: {e}", usage_metrics, state_messages, None

    def list_apps(self) -> List[Dict[str, str]]:
        """Get list of loaded apps."""
        return [
            {
                "name": app.name,
                "url": app.url if app.url else None,
                "description": app.description if app.description else None,
                "type": app.type,
            }
            for app in self.apps
        ]

    def list_tools(self) -> List[Dict[str, str]]:
        """Get list of loaded tools."""
        return [
            {"name": tool.name, "description": tool.description if tool.description else "No description"}
            for tool in self.tools
        ]


def _is_serializable(value: Any) -> bool:
    """Check if a value is serializable (can be safely stored in state context).

    Filters out modules, functions, classes, and other non-serializable objects.
    """
    # Allow basic types
    if isinstance(value, (str, int, float, bool, type(None))):
        return True

    # Allow lists and dicts (recursively check contents)
    if isinstance(value, (list, tuple)):
        return all(_is_serializable(item) for item in value)

    if isinstance(value, dict):
        return all(_is_serializable(k) and _is_serializable(v) for k, v in value.items())

    # Reject modules, functions, classes, and other non-serializable objects
    if isinstance(
        value, (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.MethodType, type)
    ):
        return False

    # Sets are not JSON serializable, so reject them
    if isinstance(value, set):
        return False

    # For other objects, try to check if they're basic types wrapped
    try:
        # Try to convert to basic types
        if hasattr(value, '__dict__'):
            # Skip objects with __dict__ (likely custom classes)
            return False
    except Exception:
        pass

    return False


async def eval_with_tools_async(code: str, _locals: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Execute code with async tools available in the local namespace."""
    original_keys = set(_locals.keys())

    try:
        with contextlib.redirect_stdout(io.StringIO()) as f:
            # Wrap the generated code in an async function and execute it
            # This allows the LLM to generate code with await statements
            # Indent the code properly - all lines including empty ones need to be indented
            indented_code = '\n'.join('    ' + line for line in code.split('\n'))
            # Check if the last line is just a variable reference and add print statement if needed
            lines = [line.strip() for line in code.split('\n') if line.strip()]
            if lines and not lines[-1].startswith(('print', 'return', '#')) and '=' not in lines[-1]:
                # Last line looks like a variable reference, add a print statement
                indented_code += f"\n    print({lines[-1]})"

            wrapped_code = f"""
import asyncio
async def __async_main():
{indented_code}
    return locals()

# Execute the wrapped function
"""
            # Create a proper global namespace with builtins and tool functions
            globals_dict = {"__builtins__": __builtins__, **_locals}
            exec(wrapped_code, globals_dict, _locals)

            # Get and run the async function
            async_main = _locals['__async_main']

            # Add timeout to prevent hanging
            result_locals = await asyncio.wait_for(async_main(), timeout=30)

            # Merge results back
            _locals.update(result_locals)

        result = f.getvalue()
        if not result:
            result = "<code ran, no output printed to stdout>"

    except asyncio.TimeoutError:
        result = "Error during execution: Execution timed out after 30 seconds"
    except Exception as e:
        result = f"Error during execution: {repr(e)}"
        import traceback

        result += f"\n{traceback.format_exc()}"

    new_keys = set(_locals.keys()) - original_keys

    # Filter out non-serializable values to avoid msgpack serialization errors
    new_vars = {}
    for key in new_keys:
        # Skip internal variables
        if key.startswith('_'):
            continue
        value = _locals[key]
        if _is_serializable(value):
            new_vars[key] = value
        else:
            logger.debug(f"Skipping non-serializable variable '{key}': {type(value).__name__}")

    # Add new variables to VariablesManager and get their preview
    if new_vars:
        var_manager = VariablesManager()
        for var_name, var_value in new_vars.items():
            # Remove old version if it exists (keep only the last one)
            # if state.variables_manager.remove_variable(var_name):
            #     logger.debug(f"Removed previous version of variable '{var_name}' from var_manager")
            # Add to variable manager
            var_manager.add_variable(var_value, name=var_name, description="Created during code execution")

        # Get formatted summary of all new variables
        try:
            variables_summary = var_manager.get_variables_summary(variable_names=list(new_vars.keys()))
            if variables_summary and variables_summary != "# No variables stored":
                result += f"\n\n## New Variables Created:\n{variables_summary}"
        except Exception as e:
            logger.debug(f"Could not generate variables summary: {e}")

    return result, new_vars


def create_mcp_prompt(
    tools,
    base_prompt=None,
    allow_user_clarification=True,
    return_to_user_cases=None,
    instructions=None,
    apps=None,
    task_loaded_from_file=False,
):
    """Create a prompt for CodeAct agent that works with MCP tools.

    Args:
        tools: List of available tools
        base_prompt: Optional base prompt to prepend
        allow_user_clarification: If True, agent can ask user for clarification. If False, only final answers allowed.
        return_to_user_cases: Optional list of custom cases (in natural language) when agent should return to user.
                             If None, uses default cases.
        instructions: Optional special instructions to include in the system prompt.
        apps: Optional list of connected apps with their descriptions
        task_loaded_from_file: If True, indicates that the task was loaded from a file
    """
    import json
    from cuga.backend.llm.utils.helpers import load_one_prompt

    processed_tools = []
    for tool in tools:
        tool_name = tool.name if hasattr(tool, 'name') else str(tool)
        tool_desc = tool.description if hasattr(tool, 'description') else "No description"

        response_schemas = {}
        if hasattr(tool, 'func') and hasattr(tool.func, '_response_schemas'):
            response_schemas = tool.func._response_schemas

        param_constraints = {}
        if hasattr(tool, 'func') and hasattr(tool.func, '_param_constraints'):
            param_constraints = tool.func._param_constraints

        if hasattr(tool, 'args_schema') and tool.args_schema:
            try:
                schema = tool.args_schema.schema()
                properties = schema.get('properties', {})
                required = schema.get('required', [])

                params = []
                for name, prop in properties.items():
                    param_type = prop.get('type', 'Any')

                    type_mapping = {
                        'string': 'str',
                        'integer': 'int',
                        'number': 'float',
                        'boolean': 'bool',
                        'array': 'list',
                        'object': 'dict',
                    }
                    python_type = type_mapping.get(param_type, param_type)

                    if name in required:
                        params.append(f"{name}: {python_type}")
                    else:
                        default_val = prop.get('default', None)
                        if default_val is not None:
                            if isinstance(default_val, str):
                                params.append(f"{name}: {python_type} = \"{default_val}\"")
                            else:
                                params.append(f"{name}: {python_type} = {default_val}")
                        else:
                            params.append(f"{name}: {python_type} = None")

                params_str = ', '.join(params) if params else ''
            except Exception as e:
                logger.debug(f"Failed to parse schema for tool {tool_name}: {e}")
                params_str = "**kwargs"
        else:
            params_str = "**kwargs"

        response_doc = ""
        if response_schemas and isinstance(response_schemas, dict):
            if 'success' in response_schemas:
                success_schema = json.dumps(response_schemas['success'], indent=4)
                response_doc += f"\n    \n    Returns (on success) - Response Schema:\n{success_schema}"

        if params_str:
            params_list = []
            if hasattr(tool, 'args_schema') and tool.args_schema:
                try:
                    schema = tool.args_schema.schema()
                    properties = schema.get('properties', {})
                    required = schema.get('required', [])

                    for name, prop in properties.items():
                        param_type = prop.get('type', 'string')
                        type_mapping = {
                            'string': 'str',
                            'integer': 'int',
                            'number': 'float',
                            'boolean': 'bool',
                            'array': 'list',
                            'object': 'dict',
                        }
                        python_type = type_mapping.get(param_type, param_type)

                        desc = prop.get('description', '')
                        required_mark = " (required)" if name in required else " (optional)"

                        constraints = param_constraints.get(name, []) or prop.get('constraints', [])
                        constraints_str = ""
                        if constraints:
                            constraints_str = f" [Constraints: {', '.join(constraints)}]"

                        params_list.append(
                            f"- `{name}`: {python_type}{required_mark} - {desc}{constraints_str}"
                        )
                except Exception:
                    params_list = [f"- {param.strip()}" for param in params_str.split(',') if param.strip()]

            params_doc = "\n".join(params_list) if params_list else "No parameters required"
        else:
            params_doc = "No parameters required"

        processed_tools.append(
            {
                'name': tool_name,
                'description': tool_desc,
                'params_str': params_str,
                'params_doc': params_doc,
                'response_doc': response_doc,
            }
        )

    processed_apps = []
    if apps:
        for app in apps:
            processed_apps.append(
                {
                    'name': app.name,
                    'type': getattr(app, 'type', 'api'),
                    'description': getattr(app, 'description', 'No description available'),
                }
            )

    prompt_template = load_one_prompt('prompts/mcp_prompt.jinja2')
    prompt = prompt_template.format(
        base_prompt=base_prompt,
        apps=processed_apps,
        allow_user_clarification=allow_user_clarification,
        return_to_user_cases=return_to_user_cases,
        instructions=instructions,
        tools=processed_tools,
        task_loaded_from_file=task_loaded_from_file,
    )
    return prompt
