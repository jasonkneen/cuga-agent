"""
Test chat message persistence across multiple turns in CugaLite.

This test verifies that chat messages are properly maintained and accumulated
across multiple conversation turns in the CugaAgent execution flow.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import CugaAgent
from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_lite_node import CugaLiteNode
from cuga.backend.cuga_graph.state.agent_state import AgentState, VariablesManager


class MockToolProvider:
    """Mock tool provider for testing."""

    async def initialize(self):
        pass

    async def get_apps(self):
        return []

    async def get_all_tools(self):
        from langchain_core.tools import tool

        @tool
        def mock_tool(query: str) -> str:
            """A mock tool for testing."""
            return f"Mock response to: {query}"

        return [mock_tool]


class TestChatMessagesPersistence:
    """Test that chat messages persist across multiple turns."""

    def setup_method(self):
        """Reset state before each test."""
        VariablesManager().reset()

    def teardown_method(self):
        """Clean up after each test."""
        VariablesManager().reset()

    @pytest.mark.asyncio
    async def test_chat_messages_returned_from_execute(self):
        """Test that execute() returns updated chat messages."""
        # Create agent with mock provider
        mock_provider = MockToolProvider()
        agent = CugaAgent(tool_provider=mock_provider)

        # Mock the agent graph to return expected state
        with patch.object(agent, 'agent') as mock_agent_graph:
            # Mock the compile method
            mock_compiled = MagicMock()
            mock_agent_graph.compile.return_value = mock_compiled

            # Simulate CodeAct agent stream returning messages
            async def mock_stream(*args, **kwargs):
                # Initial state has the user message
                initial_state = args[0] if args else kwargs.get('initial_state', {})
                messages = initial_state.get('messages', [])

                # Add AI response
                yield {
                    'messages': messages
                    + [{'role': 'assistant', 'content': 'Here is the answer to your question.'}],
                    'context': {},
                }

            mock_compiled.astream = mock_stream

            # Initialize agent
            await agent.initialize()
            agent.agent = mock_compiled

            # First turn - with previous chat history
            # Simulating a conversation where user asked "What is 2+2?" before
            previous_chat_history = [HumanMessage(content="What is 2+2?"), AIMessage(content="4")]

            answer, metrics, state_messages, updated_chat_messages = await agent.execute(
                task="Now calculate 3+3", show_progress=False, chat_messages=previous_chat_history
            )

            # Verify chat messages are returned
            assert updated_chat_messages is not None, "updated_chat_messages should not be None"
            assert len(updated_chat_messages) >= 4, (
                "Should have previous 2 messages + new user message + AI response"
            )

            # Verify message types and order
            assert isinstance(updated_chat_messages[0], HumanMessage)
            assert updated_chat_messages[0].content == "What is 2+2?"
            assert isinstance(updated_chat_messages[1], AIMessage)
            assert updated_chat_messages[1].content == "4"

            # New messages should be added
            assert isinstance(updated_chat_messages[2], HumanMessage)
            assert updated_chat_messages[2].content == "Now calculate 3+3"

            # Last message should be AI response
            assert isinstance(updated_chat_messages[-1], AIMessage)

    @pytest.mark.asyncio
    async def test_chat_messages_accumulate_across_turns(self):
        """Test that chat messages accumulate correctly across multiple turns."""
        mock_provider = MockToolProvider()
        agent = CugaAgent(tool_provider=mock_provider)

        with patch.object(agent, 'agent') as mock_agent_graph:
            mock_compiled = MagicMock()
            mock_agent_graph.compile.return_value = mock_compiled

            async def mock_stream(*args, **kwargs):
                initial_state = args[0] if args else kwargs.get('initial_state', {})
                messages = initial_state.get('messages', [])

                # Return all messages plus new AI response
                yield {
                    'messages': messages + [{'role': 'assistant', 'content': f'Response #{len(messages)}'}],
                    'context': {},
                }

            mock_compiled.astream = mock_stream

            await agent.initialize()
            agent.agent = mock_compiled

            # Turn 1 - no previous history
            _, _, _, updated_chat_turn1 = await agent.execute(
                task="Question 1",
                show_progress=False,
                chat_messages=None,  # No previous chat history
            )

            # Turn 1 should return None when chat_messages is None
            assert updated_chat_turn1 is None, "Should return None when no chat_messages provided"

            # Turn 2 - start a conversation with history from turn 1
            # Manually create history (simulating what would be stored from turn 1)
            chat_history_turn2 = [HumanMessage(content="Question 1"), AIMessage(content="Answer 1")]
            _, _, _, updated_chat_turn2 = await agent.execute(
                task="Question 2",
                show_progress=False,
                chat_messages=chat_history_turn2,  # Pass previous conversation
            )

            assert updated_chat_turn2 is not None
            assert len(updated_chat_turn2) == 4, "Should have 2 user + 2 AI messages"
            # Verify the messages are in correct order
            assert updated_chat_turn2[0].content == "Question 1"
            assert updated_chat_turn2[1].content == "Answer 1"

            # Turn 3 - continue the conversation
            _, _, _, updated_chat_turn3 = await agent.execute(
                task="Question 3",
                show_progress=False,
                chat_messages=updated_chat_turn2,  # Pass updated history from turn 2
            )

            assert updated_chat_turn3 is not None
            assert len(updated_chat_turn3) == 6, "Should have 3 user + 3 AI messages"

            # Verify message order and types
            assert isinstance(updated_chat_turn3[0], HumanMessage)
            assert isinstance(updated_chat_turn3[1], AIMessage)
            assert isinstance(updated_chat_turn3[2], HumanMessage)
            assert isinstance(updated_chat_turn3[3], AIMessage)
            assert isinstance(updated_chat_turn3[4], HumanMessage)
            assert isinstance(updated_chat_turn3[5], AIMessage)

    @pytest.mark.asyncio
    async def test_cuga_lite_node_updates_state_chat_messages(self):
        """Test that CugaLiteNode properly updates state.chat_messages."""
        node = CugaLiteNode()

        # Create initial state with no chat messages
        state = AgentState(input="Test query", url="http://example.com", final_answer="")

        # Mock the agent creation and execution
        mock_agent = AsyncMock()
        mock_agent.tools = []
        mock_agent.get_langfuse_trace_id.return_value = None

        # Simulate execute returning updated chat messages
        initial_chat = [HumanMessage(content="Previous question"), AIMessage(content="Previous answer")]
        new_chat = initial_chat + [HumanMessage(content="Test query"), AIMessage(content="Test answer")]

        mock_agent.execute.return_value = (
            "Test answer",  # answer
            {"duration_seconds": 1.0, "total_tokens": 100},  # metrics
            [],  # state_messages
            new_chat,  # updated_chat_messages
        )

        # Patch create_agent to return our mock
        with patch.object(node, 'create_agent', return_value=mock_agent):
            # Set initial chat messages in state
            state.chat_messages = initial_chat

            # Execute the node
            await node.node(state)

            # Verify state.chat_messages was updated
            assert state.chat_messages is not None
            assert len(state.chat_messages) == 4, "Should have 4 messages after update"
            assert state.chat_messages[0].content == "Previous question"
            assert state.chat_messages[1].content == "Previous answer"
            assert state.chat_messages[2].content == "Test query"
            assert state.chat_messages[3].content == "Test answer"

    @pytest.mark.asyncio
    async def test_chat_messages_none_when_not_provided(self):
        """Test that chat_messages can be None when not provided."""
        mock_provider = MockToolProvider()
        agent = CugaAgent(tool_provider=mock_provider)

        with patch.object(agent, 'agent') as mock_agent_graph:
            mock_compiled = MagicMock()
            mock_agent_graph.compile.return_value = mock_compiled

            async def mock_stream(*args, **kwargs):
                initial_state = args[0] if args else kwargs.get('initial_state', {})
                messages = initial_state.get('messages', [])

                yield {'messages': messages, 'context': {}}

            mock_compiled.astream = mock_stream

            await agent.initialize()
            agent.agent = mock_compiled

            # Execute without chat_messages (None)
            answer, metrics, state_messages, updated_chat_messages = await agent.execute(
                task="Simple query", show_progress=False, chat_messages=None
            )

            # When no chat_messages provided, should return None
            assert updated_chat_messages is None, "Should return None when chat_messages not provided"

    @pytest.mark.asyncio
    async def test_chat_messages_preserved_on_error(self):
        """Test that chat messages are handled correctly on execution error."""
        mock_provider = MockToolProvider()
        agent = CugaAgent(tool_provider=mock_provider)

        with patch.object(agent, 'agent') as mock_agent_graph:
            mock_compiled = MagicMock()
            mock_agent_graph.compile.return_value = mock_compiled

            async def mock_stream_error(*args, **kwargs):
                raise Exception("Test error")

            mock_compiled.astream = mock_stream_error

            await agent.initialize()
            agent.agent = mock_compiled

            initial_chat = [HumanMessage(content="Test question")]

            # Execute with error
            answer, metrics, state_messages, updated_chat_messages = await agent.execute(
                task="Query that will fail", show_progress=False, chat_messages=initial_chat
            )

            # Should return None for chat_messages on error
            assert updated_chat_messages is None
            assert "Error during execution" in answer
            assert metrics.get('error') is not None

    @pytest.mark.asyncio
    async def test_message_format_conversion(self):
        """Test that dict messages are properly converted to BaseMessage objects."""
        mock_provider = MockToolProvider()
        agent = CugaAgent(tool_provider=mock_provider)

        with patch.object(agent, 'agent') as mock_agent_graph:
            mock_compiled = MagicMock()
            mock_agent_graph.compile.return_value = mock_compiled

            async def mock_stream(*args, **kwargs):
                # Return messages in dict format (as CodeAct does)
                yield {
                    'messages': [
                        {'role': 'user', 'content': 'User message'},
                        {'role': 'assistant', 'content': 'AI response'},
                    ],
                    'context': {},
                }

            mock_compiled.astream = mock_stream

            await agent.initialize()
            agent.agent = mock_compiled

            # Execute with initial messages
            initial_chat = [HumanMessage(content="User message")]
            _, _, _, updated_chat = await agent.execute(
                task="Test", show_progress=False, chat_messages=initial_chat
            )

            # Verify conversion to BaseMessage objects
            assert updated_chat is not None
            assert len(updated_chat) == 2
            assert isinstance(updated_chat[0], HumanMessage)
            assert isinstance(updated_chat[1], AIMessage)
            assert updated_chat[0].content == "User message"
            assert updated_chat[1].content == "AI response"

    @pytest.mark.asyncio
    async def test_real_world_conversation_flow(self):
        """
        Integration test simulating real-world conversation flow:
        - User asks question 1
        - Agent responds
        - User asks question 2 (referring to previous context)
        - Agent responds with full conversation history
        """
        mock_provider = MockToolProvider()
        agent = CugaAgent(tool_provider=mock_provider)

        with patch.object(agent, 'agent') as mock_agent_graph:
            mock_compiled = MagicMock()
            mock_agent_graph.compile.return_value = mock_compiled

            # Track call count to generate different responses
            call_count = [0]

            async def mock_stream(*args, **kwargs):
                call_count[0] += 1
                initial_state = args[0] if args else kwargs.get('initial_state', {})
                messages = initial_state.get('messages', [])

                # Return all messages plus appropriate AI response
                if call_count[0] == 1:
                    response = "The capital of France is Paris."
                elif call_count[0] == 2:
                    response = "Yes, Paris is also known for the Eiffel Tower."
                else:
                    response = f"Response to turn {call_count[0]}"

                yield {'messages': messages + [{'role': 'assistant', 'content': response}], 'context': {}}

            mock_compiled.astream = mock_stream

            await agent.initialize()
            agent.agent = mock_compiled

            # === Turn 1: First question (no history) ===
            print("\n=== TURN 1: First question ===")
            answer1, _, _, chat_messages_after_turn1 = await agent.execute(
                task="What is the capital of France?",
                show_progress=False,
                chat_messages=None,  # No previous conversation
            )

            # Since chat_messages was None, the returned chat_messages should also be None
            assert chat_messages_after_turn1 is None
            print(f"Answer 1: {answer1}")
            print("Chat messages after turn 1: None (as expected)")

            # === Turn 2: Follow-up question (with simulated history) ===
            print("\n=== TURN 2: Follow-up question ===")
            # In real app, we'd store turn 1's conversation. Simulating that:
            conversation_history = [
                HumanMessage(content="What is the capital of France?"),
                AIMessage(content="The capital of France is Paris."),
            ]

            answer2, _, _, chat_messages_after_turn2 = await agent.execute(
                task="What is it known for?", show_progress=False, chat_messages=conversation_history
            )

            # Now chat_messages should be returned with full history
            assert chat_messages_after_turn2 is not None
            assert len(chat_messages_after_turn2) == 4, (
                f"Expected 4 messages, got {len(chat_messages_after_turn2)}"
            )

            print(f"Answer 2: {answer2}")
            print(f"Chat messages after turn 2: {len(chat_messages_after_turn2)} messages")

            # Verify conversation history is preserved
            assert chat_messages_after_turn2[0].content == "What is the capital of France?"
            assert chat_messages_after_turn2[1].content == "The capital of France is Paris."
            assert chat_messages_after_turn2[2].content == "What is it known for?"
            assert "Eiffel Tower" in chat_messages_after_turn2[3].content

            print("\n✅ Full conversation history maintained correctly!")
            print("Conversation:")
            for i, msg in enumerate(chat_messages_after_turn2):
                role = "User" if isinstance(msg, HumanMessage) else "AI"
                print(f"  {i + 1}. {role}: {msg.content[:50]}...")

    @pytest.mark.asyncio
    async def test_empty_chat_history_starts_conversation(self):
        """
        Test that passing an empty list for chat_messages starts tracking conversation history.
        This simulates the real scenario where AgentState.chat_messages defaults to [].
        """
        mock_provider = MockToolProvider()
        agent = CugaAgent(tool_provider=mock_provider)

        with patch.object(agent, 'agent') as mock_agent_graph:
            mock_compiled = MagicMock()
            mock_agent_graph.compile.return_value = mock_compiled

            async def mock_stream(*args, **kwargs):
                initial_state = args[0] if args else kwargs.get('initial_state', {})
                messages = initial_state.get('messages', [])

                yield {
                    'messages': messages + [{'role': 'assistant', 'content': 'Hello! How can I help you?'}],
                    'context': {},
                }

            mock_compiled.astream = mock_stream

            await agent.initialize()
            agent.agent = mock_compiled

            # Start with empty list (like AgentState default)
            empty_chat_history = []

            answer, metrics, state_messages, updated_chat_messages = await agent.execute(
                task="Hello",
                show_progress=False,
                chat_messages=empty_chat_history,  # Empty list, not None
            )

            # Should return updated messages even when starting from empty list
            assert updated_chat_messages is not None, "Should return messages when starting from empty list"
            assert len(updated_chat_messages) == 2, (
                f"Expected 2 messages (user + AI), got {len(updated_chat_messages)}"
            )
            assert isinstance(updated_chat_messages[0], HumanMessage)
            assert updated_chat_messages[0].content == "Hello"
            assert isinstance(updated_chat_messages[1], AIMessage)
            assert updated_chat_messages[1].content == "Hello! How can I help you?"

            print("✅ Empty chat history correctly starts conversation tracking!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print statements
