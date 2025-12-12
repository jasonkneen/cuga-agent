import pytest
from jinja2 import Template
from pathlib import Path
from cuga.backend.cuga_graph.state.agent_state import SubTaskHistory


@pytest.fixture
def user_template():
    """Load the actual user.jinja2 template"""
    template_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "cuga"
        / "backend"
        / "cuga_graph"
        / "nodes"
        / "task_decomposition_planning"
        / "plan_controller_agent"
        / "prompts"
        / "user.jinja2"
    )
    with open(template_path, 'r') as f:
        return Template(f.read())


class TestPlanControllerPrompt:
    """Test the plan controller user prompt template rendering"""

    def test_stm_history_with_final_answer(self, user_template):
        """Test rendering of stm_all_history with final_answer present"""
        context = {
            'stm_all_history': [
                SubTaskHistory(
                    sub_task='Read emails from contacts.txt',
                    steps=['Opened file', 'Extracted 7 emails'],
                    final_answer='Successfully read 7 email addresses',
                )
            ],
            'variables_history': 'No variables',
            'url': 'https://example.com',
            'input': 'Test task',
            'task_decomposition': ['Task 1', 'Task 2'],
            'current_datetime': '2025-12-12',
            'sub_tasks_progress': [],
        }

        rendered = user_template.render(context)

        assert '**Subtask 1**: Read emails from contacts.txt' in rendered
        assert '- Opened file' in rendered
        assert '- Extracted 7 emails' in rendered
        assert '**Final Answer**: Successfully read 7 email addresses' in rendered
        assert '**Final Answer**: no answer is returned' not in rendered

    def test_stm_history_without_final_answer(self, user_template):
        """Test rendering of stm_all_history when final_answer is None or empty"""
        context = {
            'stm_all_history': [
                SubTaskHistory(
                    sub_task='Search for products',
                    steps=['Navigated to catalog', 'Filtered results'],
                    final_answer=None,
                )
            ],
            'variables_history': 'No variables',
            'url': 'https://shop.com',
            'input': 'Find products',
            'task_decomposition': ['Task 1'],
            'current_datetime': '2025-12-12',
            'sub_tasks_progress': [],
        }

        rendered = user_template.render(context)

        assert '**Subtask 1**: Search for products' in rendered
        assert '- Navigated to catalog' in rendered
        assert '- Filtered results' in rendered
        assert '**Final Answer**: no answer is returned' in rendered

    def test_stm_history_empty(self, user_template):
        """Test rendering when stm_all_history is empty"""
        context = {
            'stm_all_history': [],
            'variables_history': 'No variables',
            'url': 'https://example.com',
            'input': 'Start task',
            'task_decomposition': ['Task 1', 'Task 2'],
            'current_datetime': '2025-12-12',
            'sub_tasks_progress': [],
        }

        rendered = user_template.render(context)

        assert '**Previous Subtasks**:' in rendered
        assert '**Variables History**:' in rendered

    def test_stm_history_multiple_tasks(self, user_template):
        """Test rendering with multiple completed subtasks"""
        context = {
            'stm_all_history': [
                SubTaskHistory(
                    sub_task='Find weather in London',
                    steps=['Searched weather', 'Found: 15°C, Cloudy'],
                    final_answer='London: 15°C, Cloudy',
                ),
                SubTaskHistory(
                    sub_task='Find weather in Paris',
                    steps=['Searched weather', 'Found: 18°C, Sunny'],
                    final_answer='Paris: 18°C, Sunny',
                ),
                SubTaskHistory(
                    sub_task='Compose email',
                    steps=['API call to Gmail', 'Email drafted'],
                    final_answer='',
                ),
            ],
            'variables_history': 'var_1: London weather\nvar_2: Paris weather',
            'url': 'https://weather.com',
            'input': 'Get weather and send email',
            'task_decomposition': ['Task 1', 'Task 2', 'Task 3'],
            'current_datetime': '2025-12-12',
            'sub_tasks_progress': ['completed', 'completed', 'in-progress'],
        }

        rendered = user_template.render(context)

        assert '**Subtask 1**: Find weather in London' in rendered
        assert '**Final Answer**: London: 15°C, Cloudy' in rendered
        assert '**Subtask 2**: Find weather in Paris' in rendered
        assert '**Final Answer**: Paris: 18°C, Sunny' in rendered
        assert '**Subtask 3**: Compose email' in rendered
        assert rendered.count('**Final Answer**: no answer is returned') == 1

    def test_sub_tasks_progress_display(self, user_template):
        """Test that current progress is displayed correctly"""
        context = {
            'stm_all_history': [],
            'variables_history': 'No variables',
            'url': 'https://example.com',
            'input': 'Multi-step task',
            'task_decomposition': ['Task 1', 'Task 2', 'Task 3'],
            'current_datetime': '2025-12-12',
            'sub_tasks_progress': ['completed', 'in-progress', 'not-started'],
        }

        rendered = user_template.render(context)

        assert '**Subtasks**:' in rendered
        assert '1. Task 1' in rendered
        assert '2. Task 2' in rendered
        assert '3. Task 3' in rendered

    def test_full_context_rendering(self, user_template):
        """Test full realistic scenario with all fields populated"""
        context = {
            'stm_all_history': [
                SubTaskHistory(
                    sub_task='Read email list from contacts.txt',
                    steps=['Opened file contacts.txt', 'Parsed content', 'Extracted 7 email addresses'],
                    final_answer='Successfully read email list: [user1@example.com, user2@example.com, ...]',
                )
            ],
            'variables_history': '## emails_list\n- Type: list\n- Items: 7\n- Description: Email addresses from contacts.txt',
            'url': 'file:///workspace/contacts.txt',
            'input': 'Read emails from contacts.txt and send a marketing email to each using Gmail API',
            'task_decomposition': [
                'Read the list of emails from contacts.txt (type = web, app=)',
                'For each email, compose and send marketing email (type = api, app=Gmail API)',
            ],
            'current_datetime': '2025-12-12 10:30:00',
            'sub_tasks_progress': ['completed', 'not-started'],
        }

        rendered = user_template.render(context)

        # Verify all sections are present
        assert '**Previous Subtasks**:' in rendered
        assert '**Subtask 1**: Read email list from contacts.txt' in rendered
        assert '**Final Answer**: Successfully read email list' in rendered

        assert '**Variables History**:' in rendered
        assert 'emails_list' in rendered

        assert '**Current URL**: file:///workspace/contacts.txt' in rendered

        assert '**Intent**:' in rendered
        assert 'Read emails from contacts.txt and send a marketing email' in rendered

        assert '**Subtasks**:' in rendered
        assert '1. Read the list of emails from contacts.txt' in rendered
        assert '2. For each email, compose and send marketing email' in rendered

        assert 'Current datetime: 2025-12-12 10:30:00' in rendered

    def test_stm_history_with_many_steps(self, user_template):
        """Test rendering with a subtask that has many steps"""
        context = {
            'stm_all_history': [
                SubTaskHistory(
                    sub_task='Add phones to wishlist',
                    steps=[
                        'Navigated to catalog',
                        'Clicked on Iphone 5E',
                        'Clicked Add to Wishlist',
                        'Confirmed addition',
                        'Returned to catalog',
                        'Clicked on Galaxy SE93',
                        'Clicked Add to Wishlist',
                        'Confirmed addition',
                    ],
                    final_answer='2 phones added to wishlist successfully',
                )
            ],
            'variables_history': 'phone_list: [Iphone 5E, Galaxy SE93, Xiaomi 99]',
            'url': 'https://shop.com/wishlist',
            'input': 'Add expensive phones to wishlist',
            'task_decomposition': ['Find phones', 'Add to wishlist'],
            'current_datetime': '2025-12-12',
            'sub_tasks_progress': ['completed', 'in-progress'],
        }

        rendered = user_template.render(context)

        assert '**Subtask 1**: Add phones to wishlist' in rendered
        # Check that all steps are rendered as bullet points
        assert '- Navigated to catalog' in rendered
        assert '- Clicked on Iphone 5E' in rendered
        assert '- Clicked Add to Wishlist' in rendered
        assert '**Final Answer**: 2 phones added to wishlist successfully' in rendered

    def test_special_characters_in_content(self, user_template):
        """Test that special characters are handled correctly"""
        context = {
            'stm_all_history': [
                SubTaskHistory(
                    sub_task='Search for "smartphones" & tablets',
                    steps=['Query: "smartphones" & tablets', 'Results: 10 items found'],
                    final_answer='Found 10 items matching "smartphones" & tablets',
                )
            ],
            'variables_history': 'No variables',
            'url': 'https://example.com/search?q="smartphones"&category=tablets',
            'input': 'Find "smartphones" & tablets',
            'task_decomposition': ['Search products'],
            'current_datetime': '2025-12-12',
            'sub_tasks_progress': ['completed'],
        }

        rendered = user_template.render(context)

        assert 'Search for "smartphones" & tablets' in rendered
        assert 'Query: "smartphones" & tablets' in rendered
        assert 'Found 10 items matching "smartphones" & tablets' in rendered

    def test_infinite_loop_prevention_scenario(self, user_template):
        """
        Test the exact scenario from the bug report:
        CugaLite completes a task but only appends SubTaskHistory with empty steps[]

        This validates that the template correctly handles SubTaskHistory objects
        as created by CugaLiteNode (with empty steps array)
        """
        # Simulate what CugaLiteNode actually does (line 408-414)
        context = {
            'stm_all_history': [
                SubTaskHistory(
                    sub_task='Read the list of emails from contacts.txt (type = web, app=)',
                    steps=[],  # CugaLiteNode sets this to empty array!
                    final_answer='Successfully extracted 7 emails: user1@example.com, user2@example.com, user3@example.com, user4@example.com, user5@example.com, user6@example.com, user7@example.com',
                )
            ],
            'variables_history': '## emails_list\n- Type: list\n- Items: 7\n- Description: List of email addresses from contacts.txt\n- Value: [user1@example.com, user2@example.com, ...]',
            'url': 'file:///workspace/contacts.txt',
            'input': 'Read emails from contacts.txt and send marketing email to each using Gmail API',
            'task_decomposition': [
                'Read the list of emails from contacts.txt (type = web, app=)',
                'For each email in emails_list, compose marketing email (type = api, app=Gmail API)',
                'Send each composed email (type = api, app=Gmail API)',
            ],
            'current_datetime': '2025-12-12 13:11:33',
            'sub_tasks_progress': ['completed', 'not-started', 'not-started'],
        }

        rendered = user_template.render(context)

        # Verify the controller can see the completed work
        assert '**Previous Subtasks**:' in rendered
        assert '1. Read the list of emails from contacts.txt' in rendered
        assert '**Final Answer**: Successfully extracted 7 emails' in rendered

        # When steps is empty, no steps should be rendered
        # But the final answer should still be visible
        assert 'user1@example.com' in rendered

        # Verify variables are visible
        assert '**Variables History**:' in rendered
        assert 'emails_list' in rendered

        # Verify subtasks are visible
        assert '**Subtasks**:' in rendered

    def test_cuga_lite_node_empty_steps_pattern(self, user_template):
        """
        Test exact pattern from CugaLiteNode line 408-414:
        SubTaskHistory(sub_task=state.format_subtask(), steps=[], final_answer=answer)

        This is the critical pattern that was causing the infinite loop.
        """
        # Exact pattern from CugaLiteNode
        history_entry = SubTaskHistory(
            sub_task='Read the list of emails from contacts.txt (type = web, app=)',
            steps=[],  # Always empty from CugaLiteNode!
            final_answer='Successfully read 7 email addresses from contacts.txt',
        )

        context = {
            'stm_all_history': [history_entry],
            'variables_history': '## emails_list\n- Type: list\n- Items: 7',
            'url': 'file:///workspace/contacts.txt',
            'input': 'Read emails',
            'task_decomposition': ['Read emails from file'],
            'current_datetime': '2025-12-12',
            'sub_tasks_progress': ['completed'],
        }

        rendered = user_template.render(context)

        # The subtask should be visible
        assert '**Subtask 1**: Read the list of emails from contacts.txt' in rendered

        # The final answer should be visible
        assert '**Final Answer**: Successfully read 7 email addresses' in rendered

        # No step numbers should appear (since steps=[])
        # The template has: {% for step in item['steps'] %}
        # With empty array, nothing should render in that loop
        lines = rendered.split('\n')
        subtask_section = []
        in_subtask = False
        for line in lines:
            if '1. Read the list of emails' in line:
                in_subtask = True
            elif in_subtask and '**Variables History**' in line:
                break
            elif in_subtask:
                subtask_section.append(line)

        # Should only have the final answer line, no step lines
        step_lines = [
            line for line in subtask_section if line.strip().startswith('1.') or line.strip().startswith('2.')
        ]
        assert len(step_lines) == 0, f"Expected no step lines but found: {step_lines}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
