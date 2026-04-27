import pytest
from unittest.mock import MagicMock, patch
from src.utils import JobParser, JobOfferList, JobOffer

# Fixture to mock the config module
@pytest.fixture
def mock_config():
    with patch('src.utils.config') as mock_cfg:
        mock_cfg.openrouter_base_url = "http://mock-openrouter.com"
        mock_cfg.openrouter_api_key = "mock-key"
        yield mock_cfg

# Fixture to mock ChatOpenAI
@pytest.fixture
def mock_chat_openai():
    with patch('src.utils.ChatOpenAI') as mock_llm_class:
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance
        yield mock_llm_class, mock_llm_instance

# --- Tests for JobParser ---

def test_job_parser_initialization(mock_config, mock_chat_openai):
    mock_llm_class, mock_llm_instance = mock_chat_openai

    parser = JobParser(model_name="test-model")

    mock_llm_class.assert_called_once_with(
        model="test-model",
        temperature=0,
        base_url=mock_config.openrouter_base_url,
        api_key=mock_config.openrouter_api_key,
    )
    mock_llm_instance.with_structured_output.assert_called_once_with(JobOfferList)
    assert parser.llm == mock_llm_instance
    assert parser.structured_llm == mock_llm_instance.with_structured_output.return_value

def test_job_parser_parse_success(mock_config, mock_chat_openai):
    _, mock_llm_instance = mock_chat_openai

    # Setup mock for structured_llm.invoke
    mock_structured_llm_invoke = MagicMock()
    mock_llm_instance.with_structured_output.return_value.invoke = mock_structured_llm_invoke

    # Define the expected output from the LLM
    expected_job_offers = [
        JobOffer(id="comp-title-1", title="Dev", company="CompA", location="NY", salary="100k", description="Desc1", url="url1"),
        JobOffer(id="comp-title-2", title="Eng", company="CompB", location="CA", salary="N/A", description="Desc2", url="url2"),
    ]
    mock_structured_llm_invoke.return_value = JobOfferList(offers=expected_job_offers)

    parser = JobParser()
    raw_search_results = "Some raw text with job postings."
    parsed_jobs = parser.parse(raw_search_results)

    # Assert invoke was called with correct prompts
    mock_structured_llm_invoke.assert_called_once()
    args, kwargs = mock_structured_llm_invoke.call_args
    assert len(args[0]) == 2 # system and user messages
    assert args[0][0]["role"] == "system"
    assert "expert Data Extraction Agent" in args[0][0]["content"]
    assert args[0][1]["role"] == "user"
    assert raw_search_results in args[0][1]["content"]

    # Assert the returned jobs match the expected ones
    assert parsed_jobs == expected_job_offers

def test_job_parser_parse_error_handling(mock_config, mock_chat_openai, capsys):
    _, mock_llm_instance = mock_chat_openai

    mock_structured_llm_invoke = MagicMock()
    mock_llm_instance.with_structured_output.return_value.invoke = mock_structured_llm_invoke
    mock_structured_llm_invoke.side_effect = Exception("LLM parsing failed")

    parser = JobParser()
    parsed_jobs = parser.parse("Some raw text")

    assert parsed_jobs == []
    captured = capsys.readouterr()
    assert "❌ Error during job parsing: LLM parsing failed" in captured.out