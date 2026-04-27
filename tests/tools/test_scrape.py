import pytest
from unittest.mock import patch, Mock
import requests # Import requests to access its exceptions
from src.tools.scrape import scrape_webpage_tool

class TestScrapeWebpageTool:
    """
    Tests for the scrape_webpage_tool function.
    """

    @pytest.fixture
    def mock_response(self):
        """Fixture to create a mock requests.Response object."""
        mock = Mock()
        mock.status_code = 200
        mock.raise_for_status.return_value = None
        return mock

    def test_successful_scrape(self, mock_response):
        """
        Tests that scrape_webpage_tool successfully fetches, cleans, and returns
        the text content of a webpage.
        """
        test_url = "http://example.com/job"
        mock_response.text = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Job Title</h1>
                <p>This is a job description.</p>
                <script>alert('malicious script');</script>
                <style>.hidden { display: none; }</style>
                <div>More <span class="highlight">details</span> here.</div>
            </body>
        </html>
        """
        expected_text = "Test Page\nJob Title\nThis is a job description.\nMore\ndetails\nhere."

        with patch('requests.get', return_value=mock_response) as mock_get:
            result = scrape_webpage_tool._run(test_url, config={})

            mock_get.assert_called_once_with(
                test_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"},
                timeout=10
            )
            assert result == expected_text
            assert len(result) <= 10000

    def test_api_connection_error(self):
        """
        Tests that scrape_webpage_tool handles connection errors gracefully.
        """
        test_url = "http://nonexistent.com"
        with patch('requests.get', side_effect=requests.exceptions.RequestException("Connection refused")) as mock_get:
            result = scrape_webpage_tool._run(test_url, config={})

            mock_get.assert_called_once_with(
                test_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"},
                timeout=10
            )
            assert f"Error fetching webpage {test_url}" in result
            assert "Connection refused" in result

    def test_http_error(self, mock_response):
        """
        Tests that scrape_webpage_tool handles HTTP errors (e.g., 404, 500) gracefully.
        """
        test_url = "http://example.com/404"
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "404 Client Error: Not Found for url: http://example.com/404"
        )

        with patch('requests.get', return_value=mock_response) as mock_get:
            result = scrape_webpage_tool._run(test_url, config={})

            mock_get.assert_called_once_with(
                test_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"},
                timeout=10
            )
            assert f"Error fetching webpage {test_url}" in result
            assert "404 Client Error" in result

    def test_empty_content(self, mock_response):
        """
        Tests that scrape_webpage_tool returns an empty string for a page with no text content.
        """
        test_url = "http://example.com/empty"
        mock_response.text = "<html><body><script></script><style></style></body></html>"
        with patch('requests.get', return_value=mock_response):
            result = scrape_webpage_tool._run(test_url, config={})
            assert result == ""

    def test_truncation(self, mock_response):
        """
        Tests that the output is truncated to 10000 characters.
        """
        test_url = "http://example.com/long"
        long_text = "a" * 15000 # Create a string longer than 10000
        mock_response.text = f"<html><body>{long_text}</body></html>"

        with patch('requests.get', return_value=mock_response):
            result = scrape_webpage_tool._run(test_url, config={})
            assert len(result) == 10000
            assert result == "a" * 10000