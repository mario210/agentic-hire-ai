import pytest
from unittest.mock import patch, Mock
import requests # Import requests to access its exceptions
from src.tools.search import job_search_tool, JobSearchProvider

# Mock the config object globally for these tests
@pytest.fixture(autouse=True)
def mock_config():
    """
    Fixture to mock the application configuration, specifically the OrioSearch base URL.
    This ensures tests don't make actual external calls and use a predictable URL.
    """
    with patch('src.tools.search.config') as mock_app_config:
        mock_app_config.oriosearch_base_url = "http://mock-oriosearch.com/api"
        yield mock_app_config

class TestJobSearchTool:
    """
    Tests for the job_search_tool function.
    """

    def test_successful_search(self, mock_config):
        """
        Tests that job_search_tool successfully makes an API call and returns
        the search results as a string.
        """
        mock_response_data = {
            "results": [
                {"title": "Job 1", "snippet": "Desc 1", "link": "http://link1.com"},
                {"title": "Job 2", "snippet": "Desc 2", "link": "http://link2.com"},
            ]
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None # Indicate no HTTP error

        with patch('requests.post', return_value=mock_response) as mock_post:
            query = "Python Developer jobs Poland"
            result = job_search_tool._run(query, config={})

            mock_post.assert_called_once_with(
                mock_config.oriosearch_base_url,
                json={"query": query, "num_results": 10},
                timeout=10
            )
            assert result == str(mock_response_data)

    def test_api_connection_error(self, mock_config):
        """
        Tests that job_search_tool handles connection errors gracefully.
        """
        with patch('requests.post', side_effect=requests.exceptions.RequestException("Connection refused")) as mock_post:
            query = "Data Scientist jobs remote"
            result = job_search_tool._run(query, config={})

            mock_post.assert_called_once_with(
                mock_config.oriosearch_base_url,
                json={"query": query, "num_results": 10},
                timeout=10
            )
            assert "Error connecting to OrioSearch" in result
            assert "Connection refused" in result

    def test_http_error(self, mock_config):
        """
        Tests that job_search_tool handles HTTP errors (e.g., 404, 500) gracefully,
        as they are caught by RequestException.
        """
        mock_response = Mock()
        mock_response.status_code = 404
        # raise_for_status is called before json(), so it will raise an error
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "404 Client Error: Not Found for url: http://mock-oriosearch.com/api"
        )

        with patch('requests.post', return_value=mock_response) as mock_post:
            query = "Software Engineer jobs New York"
            result = job_search_tool._run(query, config={})

            mock_post.assert_called_once_with(
                mock_config.oriosearch_base_url,
                json={"query": query, "num_results": 10},
                timeout=10
            )
            assert "Error connecting to OrioSearch" in result
            assert "404 Client Error" in result


class TestJobSearchProvider:
    """
    Tests for the JobSearchProvider class.
    """

    def test_initialization(self):
        """
        Tests that the JobSearchProvider correctly initializes and
        assigns the job_search_tool.
        """
        provider = JobSearchProvider()
        assert provider.search_tool is job_search_tool