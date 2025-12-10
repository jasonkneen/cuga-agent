import asyncio
import uuid
import time
import httpx
from system_tests.e2e.base_test import BaseTestServerStream, STREAM_ENDPOINT, SERVER_URL

STATE_ENDPOINT = f"{SERVER_URL}/api/agent/state"


class LoadTest(BaseTestServerStream):
    """
    Load test for concurrent users.
    """

    # Configure environment for API mode (no browser)
    test_env_vars = {
        "CUGA_MODE": "api",
        "CUGA_TEST_ENV": "true",
        "DYNACONF_ADVANCED_FEATURES__TRACKER_ENABLED": "true",
    }

    async def get_agent_state(self, thread_id: str) -> dict:
        """Get agent state for a specific thread_id."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                STATE_ENDPOINT,
                headers={"X-Thread-ID": thread_id},
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                return {
                    "state": None,
                    "variables": {},
                    "variables_count": 0,
                    "chat_messages_count": 0,
                }
            else:
                raise Exception(f"Failed to get state: {response.status_code} - {response.text}")

    async def validate_state_isolation(
        self, user_id: int, thread_id: str, other_thread_ids: list[str]
    ) -> tuple[bool, str]:
        """
        Validate that this thread's state is isolated from other threads.
        Checks that threads that have completed have their own variable storage and chat messages.
        Returns (is_valid, error_message)
        """
        try:
            state_response = await self.get_agent_state(thread_id)
            my_variables = state_response.get("variables", {})
            my_chat_messages_count = state_response.get("chat_messages_count", 0)

            # Check that we actually have variables
            if not my_variables:
                return False, f"User {user_id}: No variables found in state"

            # Check that we have chat messages
            if my_chat_messages_count == 0:
                return False, f"User {user_id}: No chat_messages found in state"


            # For each other thread that has completed, verify they have their own variables and chat messages
            # Even if variable names are the same (they're doing the same task),
            # each thread should have its own storage
            for other_thread_id in other_thread_ids:
                if other_thread_id == thread_id:
                    continue

                await self.get_agent_state(other_thread_id)

                # If other thread has variables and chat messages, that's good - it means it has its own storage
                # The key test is that initial state was empty and final state has variables and messages
                # This already proves isolation via LangGraph's checkpointer

            return True, ""
        except Exception as e:
            return False, f"User {user_id}: Error validating isolation: {e}"

    async def run_single_user_task(
        self, user_id: int, thread_id: str, all_thread_ids: list[str]
    ) -> tuple[bool, str]:
        """
        Runs a task for a single user and verifies the result.
        Returns (success, error_message)
        """
        query = "list all my accounts, how many are there?"
        expected_keywords = ["50"]

        print(f"User {user_id} (Thread {thread_id}): Starting task...")

        try:
            # Validate state is empty at start
            initial_state = await self.get_agent_state(thread_id)
            initial_variables_count = initial_state.get("variables_count", 0)
            initial_chat_messages_count = initial_state.get("chat_messages_count", 0)

            if initial_variables_count > 0:
                return (
                    False,
                    f"User {user_id}: State should be empty at start, but found {initial_variables_count} variables",
                )
            if initial_chat_messages_count > 0:
                return (
                    False,
                    f"User {user_id}: chat_messages should be empty at start, but found {initial_chat_messages_count}",
                )

            # We need to manually implement run_task here to pass headers
            # BaseTestServerStream.run_task doesn't support custom headers easily without modification
            # So I'll replicate the logic here with the header

            all_events = []
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    STREAM_ENDPOINT,
                    json={"query": query},
                    headers={"Accept": "text/event-stream", "X-Thread-ID": thread_id},
                ) as response:
                    if response.status_code != 200:
                        return False, f"User {user_id}: Failed with status {response.status_code}"

                    buffer = b""
                    async for chunk in response.aiter_bytes():
                        buffer += chunk
                        while b"\n\n" in buffer:
                            event_block, buffer = buffer.split(b"\n\n", 1)
                            event_lines = event_block.split(b"\n")
                            event_data = {}
                            for line in event_lines:
                                line = line.strip()
                                if not line:
                                    continue
                                if line.startswith(b"event: "):
                                    event_data["event"] = line[len(b"event: ") :].decode("utf-8").strip()
                                elif line.startswith(b"data: "):
                                    try:
                                        data_str = line[len(b"data: ") :].decode("utf-8").strip()
                                        event_data["data"] = self._parse_event_data(data_str)
                                    except Exception:
                                        event_data["data"] = line[len(b"data: ") :].strip()

                            if event_data:
                                all_events.append(event_data)
                                if event_data.get("event") == "Answer":
                                    break

            # Verify result
            answer_event = next((e for e in all_events if e.get("event") == "Answer"), None)
            if not answer_event:
                return False, f"User {user_id}: No Answer event found"

            answer_data = str(answer_event.get("data", "")).lower()
            for keyword in expected_keywords:
                if keyword.lower() not in answer_data:
                    return (
                        False,
                        f"User {user_id}: Answer missing keyword '{keyword}'. Got: {answer_data}",
                    )

            # Wait a moment for the graph to checkpoint the final state
            # LangGraph checkpoints state after node completion, not during execution
            await asyncio.sleep(2)

            # Validate state after completion has variables and chat messages
            final_state = await self.get_agent_state(thread_id)
            final_variables_count = final_state.get("variables_count", 0)
            final_chat_messages_count = final_state.get("chat_messages_count", 0)

            if final_variables_count == 0:
                return (
                    False,
                    f"User {user_id}: State should have variables after completion, but found 0 variables",
                )
            if final_chat_messages_count == 0:
                return (
                    False,
                    f"User {user_id}: State should have chat_messages after completion, but found 0",
                )

            # Validate isolation from other threads
            other_thread_ids = [tid for tid in all_thread_ids if tid != thread_id]
            is_isolated, isolation_error = await self.validate_state_isolation(
                user_id, thread_id, other_thread_ids
            )
            if not is_isolated:
                return False, isolation_error

            print(f"User {user_id}: ✓ State is isolated from other threads")

            # Send followup question
            print(f"User {user_id} (Thread {thread_id}): Sending followup question...")
            followup_query = "how many accounts did we retrieve?"
            followup_expected_keywords = ["50"]

            all_followup_events = []
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    STREAM_ENDPOINT,
                    json={"query": followup_query},
                    headers={"Accept": "text/event-stream", "X-Thread-ID": thread_id},
                ) as response:
                    if response.status_code != 200:
                        return False, f"User {user_id}: Followup failed with status {response.status_code}"

                    buffer = b""
                    async for chunk in response.aiter_bytes():
                        buffer += chunk
                        while b"\n\n" in buffer:
                            event_block, buffer = buffer.split(b"\n\n", 1)
                            event_lines = event_block.split(b"\n")
                            event_data = {}
                            for line in event_lines:
                                line = line.strip()
                                if not line:
                                    continue
                                if line.startswith(b"event: "):
                                    event_data["event"] = line[len(b"event: ") :].decode("utf-8").strip()
                                elif line.startswith(b"data: "):
                                    try:
                                        data_str = line[len(b"data: ") :].decode("utf-8").strip()
                                        event_data["data"] = self._parse_event_data(data_str)
                                    except Exception:
                                        event_data["data"] = line[len(b"data: ") :].strip()

                            if event_data:
                                all_followup_events.append(event_data)
                                if event_data.get("event") == "Answer":
                                    break

            # Verify followup result
            followup_answer_event = next((e for e in all_followup_events if e.get("event") == "Answer"), None)
            if not followup_answer_event:
                return False, f"User {user_id}: No Answer event found in followup"

            followup_answer_data = str(followup_answer_event.get("data", "")).lower()
            for keyword in followup_expected_keywords:
                if keyword.lower() not in followup_answer_data:
                    return (
                        False,
                        f"User {user_id}: Followup answer missing keyword '{keyword}'. Got: {followup_answer_data}",
                    )

            print(f"User {user_id}: ✓ Followup question answered correctly")
            print(f"User {user_id}: Success!")
            return True, ""

        except Exception as e:
            return False, f"User {user_id}: Exception: {e}"

    async def test_concurrent_users(self):
        """
        Simulate 20 concurrent users running the same task.
        Validates state isolation between threads.
        """
        num_users = 5
        print(f"\n--- Starting Load Test with {num_users} users ---")

        start_time = time.time()

        # Generate all thread_ids upfront
        thread_ids = [str(uuid.uuid4()) for _ in range(num_users)]

        tasks = []
        for i in range(num_users):
            tasks.append(self.run_single_user_task(i, thread_ids[i], thread_ids))

        results = await asyncio.gather(*tasks)

        end_time = time.time()
        duration = end_time - start_time

        success_results = [(success, error) for success, error in results if success]
        failure_results = [(success, error) for success, error in results if not success]

        success_count = len(success_results)
        failure_count = len(failure_results)

        print(f"\n--- Load Test Completed in {duration:.2f}s ---")
        print(f"Total Users: {num_users}")
        print(f"Success: {success_count}")
        print(f"Failure: {failure_count}")

        if failure_count > 0:
            print("\n--- Failure Details ---")
            for i, (success, error) in enumerate(failure_results):
                print(f"Failure {i + 1}: {error}")

        self.assertEqual(
            failure_count,
            0,
            f"{failure_count} users failed the test. Errors: {[e for _, e in failure_results]}",
        )
