"""Tests for context strategy agent."""
import asyncio
import pytest
from sik_llms import RegisteredClients
from server.agents.context_strategy_agent import (
    ContextStrategyAgent,
    ContextType,
)
from dotenv import load_dotenv
load_dotenv()

TEST_MODEL = 'gpt-4o'
VALID_STRATEGIES = [ContextType.IGNORE, ContextType.FULL_TEXT, ContextType.RAG]


@pytest.mark.asyncio
class TestEvalRetrievalStrategies:
    """
    Evaluate different retrieval strategies based on file types and questions.

    TODO: Move these "evals" out of unit tests and into a separate evaluation script.
    """

    @pytest.fixture
    def agent(self):
        """Create a basic agent."""
        model_config = {
            'client_type': RegisteredClients.OPENAI,
            'model_name': TEST_MODEL,
        }
        return ContextStrategyAgent(**model_config)

    @pytest.mark.parametrize('test_case', [
        pytest.param(
            {
                # Server API question should use server file and ignore UI
                'question': 'Show the implementation in the server for api requests.',
                'files': ['server_api.py', 'client_ui.tsx', 'client.css', 'attention_is_all_you_need.pdf'],  # noqa: E501
                'expected_strategies': {
                    'server_api.py': [ContextType.FULL_TEXT, ContextType.RAG],
                    'client_ui.tsx': [ContextType.IGNORE],
                    'client.css': [ContextType.IGNORE],
                    'attention_is_all_you_need.pdf': [ContextType.IGNORE],
                },
            },
            id='server_api_handling',
        ),
        pytest.param(
            {
                # UI question should use UI file and ignore server
                'question': 'Show the formatting for the login form in the UI files?',
                'files': ['client_login_ui.tsx', 'grpc_api_service.py', 'main.css', 'attention_is_all_you_need.pdf'],  # noqa: E501
                'expected_strategies': {
                    'client_login_ui.tsx': [ContextType.FULL_TEXT, ContextType.RAG],
                    'grpc_api_service.py': [ContextType.IGNORE],
                    'main.css': [ContextType.FULL_TEXT, ContextType.RAG],
                    'attention_is_all_you_need.pdf': [ContextType.IGNORE],
                },
            },
            id='ui_implementation',
        ),
    ])
    async def test_code_file_handling(
            self,
            agent: ContextStrategyAgent,
            test_case: dict,
        ):
        """Test handling of code files with multiple runs."""
        sample_size = 10
        pass_threshold = 5
        messages = [{"role": "user", "content": test_case["question"]}]
        files = test_case["files"]
        expected_strategies = test_case["expected_strategies"]
        # Run n times concurrently
        summaries = await asyncio.gather(*(
            agent(messages=messages, resource_names=files)
            for _ in range(sample_size)
        ))
        # [s.strategies[i].resource_name for s in summaries]
        # [s.strategies[i].context_type for s in summaries]
        for i, (expected_name, expected_strategy) in enumerate(expected_strategies.items()):
            assert sum(s.strategies[i].resource_name == expected_name for s in summaries) >= pass_threshold  # noqa: E501
            assert sum(s.strategies[i].context_type in expected_strategy for s in summaries) >= pass_threshold  # noqa: E501


    @pytest.mark.parametrize('test_case', [
        pytest.param(
            {
                # UI question should use UI file and ignore server
                'question': 'Give a code review for `grpc_api_service.py`.',
                'files': ['client_login_ui.tsx', 'grpc_api_service.py', 'main.css', 'attention_is_all_you_need.pdf'],  # noqa: E501
                'expected_strategies': {
                    'client_login_ui.tsx': [ContextType.IGNORE],
                    'grpc_api_service.py': [ContextType.FULL_TEXT, ContextType.RAG],
                    'main.css': [ContextType.IGNORE],
                    'attention_is_all_you_need.pdf': [ContextType.IGNORE],
                },
            },
            id='grpc_api_service',
        ),
    ])
    async def test_code_file_handling__file_name_is_mentioned_by_user(
            self,
            test_case: dict,
            agent: ContextStrategyAgent,
        ):
        """Test handling of code files with multiple runs."""
        sample_size = 10
        pass_threshold = 5
        messages = [{"role": "user", "content": test_case["question"]}]
        files = test_case["files"]
        expected_strategies = test_case["expected_strategies"]
        summaries = await asyncio.gather(*(
            agent(messages=messages, resource_names=files)
            for _ in range(sample_size)
        ))
        # [s.strategies[i].resource_name for s in summaries]
        # [s.strategies[i].context_type for s in summaries]
        for i, (expected_name, expected_strategy) in enumerate(expected_strategies.items()):
            assert sum(s.strategies[i].resource_name == expected_name for s in summaries) >= pass_threshold  # noqa: E501
            assert sum(s.strategies[i].context_type in expected_strategy for s in summaries) >= pass_threshold  # noqa: E501

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize('test_case', [
        pytest.param(
            {
                'question': 'Please provide a summary of all financial documents.',
                'files': ['2023_annual_report.pdf', '2024_q1_report.pdf', 'server_api.py', 'weather_report.pdf'],  # noqa: E501
                'expected_strategies': {
                    '2023_annual_report.pdf': [ContextType.FULL_TEXT],
                    '2024_q1_report.pdf': [ContextType.FULL_TEXT],
                    'server_api.py': [ContextType.IGNORE],
                    'weather_report.pdf': [ContextType.IGNORE],
                },
            },
            id='summarize_documents',
        ),
        pytest.param(
            {
                'question': 'What were the Q3 2023 revenue numbers for the North America region?',
                'files': ['2023_annual_report.pdf', '2023_q3_report.pdf', 'q3_meeting_notes.txt', 'weather_report.md'],  # noqa: E501
                'expected_strategies': {
                    # the model might ignore annual because we have the q3 report, or it might use
                    '2023_annual_report.pdf': [ContextType.IGNORE, ContextType.RAG, ContextType.FULL_TEXT],  # noqa: E501
                    # the answer is likey found in a specific section of the q3 report
                    '2023_q3_report.pdf': [ContextType.RAG],
                    'q3_meeting_notes.txt': [ContextType.IGNORE, ContextType.RAG, ContextType.FULL_TEXT],  # noqa: E501
                    'weather_report.md': [ContextType.IGNORE],
                },
            },
            id='specific_information_query',
        ),
        pytest.param(
            {
                'question': "What is attention in a transfomer?",
                'files': [
                    'attention_is_all_you_need.pdf',
                    'financial_summary.pdf',
                    'q4_projections.xlsx',
                    'strategic_plan.docx',
                ],
                'expected_strategies': {
                    'attention_is_all_you_need.pdf': [ContextType.RAG],
                    'financial_summary.pdf': [ContextType.IGNORE],
                    'q4_projections.xlsx': [ContextType.IGNORE],
                    'strategic_plan.docx': [ContextType.IGNORE],
                },
            },
            id='attention',
        ),
    ])
    async def test_text_file_strategies(
            self,
            agent: ContextStrategyAgent,
            test_case: dict,
        ):
        """Test different strategies for text files."""
        sample_size = 10
        pass_threshold = 5
        messages = [{"role": "user", "content": test_case["question"]}]
        files = test_case["files"]
        expected_strategies = test_case["expected_strategies"]
        summaries = await asyncio.gather(*(
            agent(messages=messages, resource_names=files)
            for _ in range(sample_size)
        ))
        # [s.strategies[i].resource_name for s in summaries]
        # [s.strategies[i].context_type for s in summaries]
        # [s.strategies[i].reasoning for s in summaries]
        for i, (expected_name, expected_strategy) in enumerate(expected_strategies.items()):
            assert sum(s.strategies[i].resource_name == expected_name for s in summaries) >= pass_threshold  # noqa: E501
            assert sum(s.strategies[i].context_type in expected_strategy for s in summaries) >= pass_threshold  # noqa: E501
