"""Tests for context strategy agent."""
import asyncio
import pytest
from server.agents.context_strategy_agent import (
    ContextStrategyAgent,
    ContextStrategyResult,
    ContextStrategySummary,
    ContextType,
)
from dotenv import load_dotenv
load_dotenv()

TEST_MODEL = "gpt-4o-mini"
VALID_STRATEGIES = [ContextType.IGNORE, ContextType.FULL_TEXT, ContextType.RAG]

@pytest.mark.asyncio
class TestContextStrategyBasics:
    """Test basic functionality of the ContextStrategyAgent."""

    @pytest.fixture
    def agent(self):
        """Create a basic agent."""
        return ContextStrategyAgent(model=TEST_MODEL)

    async def test_single_file_summary(self, agent: ContextStrategyAgent):
        """Test getting strategy for a single file."""
        messages = [
            {"role": "user", "content": "Can you summarize this document?"},
        ]
        files = ['report.pdf']
        summary = await agent(messages=messages, resource_names=files)
        assert isinstance(summary, ContextStrategySummary)
        assert len(summary.strategies) == 1
        assert isinstance(summary.strategies[0], ContextStrategyResult)
        assert summary.strategies[0].resource_name == files[0]
        assert summary.strategies[0].context_type in VALID_STRATEGIES
        assert summary.strategies[0].reasoning
        assert summary.total_input_tokens > 0
        assert summary.total_output_tokens > 0
        assert summary.total_input_cost > 0
        assert summary.total_output_cost > 0
        assert summary.total_cost == pytest.approx(summary.total_input_cost + summary.total_output_cost)  # noqa: E501

    async def test_multiple_files_summary(self, agent: ContextStrategyAgent):
        """Test getting strategy for multiple files."""
        messages = [
            {"role": "user", "content": "What's the total revenue mentioned in these reports?"},
        ]
        files = ["q1_report.pdf", "q2_report.pdf", "q3_report.pdf"]
        summary = await agent(messages=messages, resource_names=files)
        assert len(summary.strategies) == 3
        assert summary.total_cost == pytest.approx(summary.total_input_cost + summary.total_output_cost)  # noqa: E501
        for result, resource_name in zip(summary.strategies, files, strict=True):
            assert result.resource_name == resource_name
            assert result.context_type in VALID_STRATEGIES
            assert result.reasoning


@pytest.mark.asyncio
class TestRetrievalStrategies:
    """Test different retrieval strategies based on file types and questions."""

    @pytest.fixture
    def agent(self):
        """Create a basic agent."""
        return ContextStrategyAgent(model=TEST_MODEL)

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
        # Run 20 times concurrently
        summaries = await asyncio.gather(*(
            agent(messages=messages, resource_names=files)
            for _ in range(sample_size)
        ))
        # [s.results[1].arguments for s in summaries]
        for i in range(len(expected_strategies)):
            assert sum(s.strategies[i].resource_name in expected_strategies for s in summaries) >= pass_threshold  # noqa: E501
            assert sum(s.strategies[i].context_type in expected_strategies[s.strategies[i].resource_name] for s in summaries) >= pass_threshold  # noqa: E501

    @pytest.mark.parametrize('test_case', [
        pytest.param(
            {
                'question': 'Please provide a summary of these financial documents.',
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
                    '2023_annual_report.pdf': [ContextType.RAG],
                    '2023_q3_report.pdf': [ContextType.RAG],
                    'q3_meeting_notes.txt': [ContextType.RAG],
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
        # [s.strategies[0] for s in summaries]
        for i in range(len(expected_strategies)):
            assert sum(s.strategies[i].resource_name in expected_strategies for s in summaries) >= pass_threshold  # noqa: E501
            assert sum(s.strategies[i].context_type in expected_strategies[s.strategies[i].resource_name] for s in summaries) >= pass_threshold  # noqa: E501
