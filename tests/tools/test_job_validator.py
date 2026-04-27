import pytest
from unittest.mock import patch, Mock
import requests
from src.tools.job_validator import JobValidator, JobOffer, ExpirationCheck

class TestJobValidator:
    """
    Tests for the JobValidator class.
    """

    @pytest.fixture
    def mock_llm(self):
        """Fixture to create a mock LLM instance."""
        mock_llm_instance = Mock()
        # Mock the with_structured_output method to return a mock checker
        mock_llm_instance.with_structured_output.return_value = Mock()
        return mock_llm_instance

    @pytest.fixture
    def validator(self, mock_llm):
        """Fixture to create a JobValidator instance with a mocked LLM."""
        return JobValidator(mock_llm)

    @pytest.fixture
    def mock_requests_get(self):
        """Fixture to patch requests.get."""
        with patch('requests.get') as mock_get:
            yield mock_get

    @pytest.fixture
    def mock_checker_invoke(self, validator):
        """Fixture to patch the checker.invoke method."""
        with patch.object(validator.checker, 'invoke') as mock_invoke:
            yield mock_invoke

    def test_initialization(self, mock_llm):
        """
        Tests that the JobValidator initializes correctly and sets up the checker.
        """
        validator = JobValidator(mock_llm)
        mock_llm.with_structured_output.assert_called_once_with(ExpirationCheck)
        assert validator.checker is not None

    @pytest.mark.parametrize("invalid_url", [
        "N/A",
        "ftp://example.com",
        "not-a-url"
    ])
    def test_invalid_url_format(self, validator, invalid_url):
        """
        Tests that is_job_valid returns False for invalid URL formats.
        """
        job = JobOffer(id="test-id", title="Test Job", url=invalid_url, company="TestCo", location="Remote")
        assert not validator.is_job_valid(job)

    def test_http_error_status_code(self, validator, mock_requests_get, mock_checker_invoke):
        """
        Tests that is_job_valid returns False when requests.get returns an HTTP error status.
        """
        mock_response = Mock()
        mock_response.status_code = 404
        mock_requests_get.return_value = mock_response

        job = JobOffer(id="test-id", title="Test Job", url="http://example.com/404", company="TestCo", location="Remote")
        assert not validator.is_job_valid(job)
        mock_requests_get.assert_called_once()
        mock_checker_invoke.assert_not_called() # LLM should not be called on HTTP error

    def test_request_exception(self, validator, mock_requests_get, mock_checker_invoke):
        """
        Tests that is_job_valid returns False when requests.get raises a RequestException.
        """
        mock_requests_get.side_effect = requests.exceptions.RequestException("Connection refused")

        job = JobOffer(id="test-id", title="Test Job", url="http://example.com/timeout", company="TestCo", location="Remote")
        assert not validator.is_job_valid(job)
        mock_requests_get.assert_called_once()
        mock_checker_invoke.assert_not_called() # LLM should not be called on request exception

    def test_llm_determines_inactive(self, validator, mock_requests_get, mock_checker_invoke):
        """
        Tests that is_job_valid returns False when the LLM determines the job is inactive.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>Job closed. Applications no longer accepted.</body></html>"
        mock_requests_get.return_value = mock_response

        mock_checker_invoke.return_value = ExpirationCheck(is_active=False, reason="Job is closed")

        job = JobOffer(id="test-id", title="Test Job", url="http://example.com/closed", company="TestCo", location="Remote")
        assert not validator.is_job_valid(job)
        mock_requests_get.assert_called_once()
        mock_checker_invoke.assert_called_once()

    def test_llm_determines_active(self, validator, mock_requests_get, mock_checker_invoke):
        """
        Tests that is_job_valid returns True when the LLM determines the job is active.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>Apply now for this exciting opportunity!</body></html>"
        mock_requests_get.return_value = mock_response

        mock_checker_invoke.return_value = ExpirationCheck(is_active=True, reason="Job is active")

        job = JobOffer(id="test-id", title="Test Job", url="http://example.com/active", company="TestCo", location="Remote")
        assert validator.is_job_valid(job)
        mock_requests_get.assert_called_once()
        mock_checker_invoke.assert_called_once()

    def test_general_exception_during_validation(self, validator, mock_requests_get, mock_checker_invoke):
        """
        Tests that is_job_valid returns False for unexpected exceptions during the process.
        """
        # Simulate an error during BeautifulSoup parsing or text extraction
        mock_response = Mock()
        mock_response.status_code = 200
        # Malformed HTML that might cause BeautifulSoup to raise an error, or just a generic error
        mock_response.text = "<html><body"
        mock_requests_get.return_value = mock_response

        # We'll mock the internal BeautifulSoup.get_text to raise an error
        with patch('bs4.BeautifulSoup.get_text', side_effect=Exception("Parsing error")):
            job = JobOffer(id="test-id", title="Test Job", url="http://example.com/malformed", company="TestCo", location="Remote")
            assert not validator.is_job_valid(job)
            mock_requests_get.assert_called_once()
            mock_checker_invoke.assert_not_called() # LLM should not be called if parsing fails

    def test_llm_invoke_exception(self, validator, mock_requests_get, mock_checker_invoke):
        """
        Tests that is_job_valid returns False if the LLM invocation fails.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>Some job content.</body></html>"
        mock_requests_get.return_value = mock_response

        mock_checker_invoke.side_effect = Exception("LLM service unavailable")

        job = JobOffer(id="test-id", title="Test Job", url="http://example.com/llm-fail", company="TestCo", location="Remote")
        assert not validator.is_job_valid(job)
        mock_requests_get.assert_called_once()
        mock_checker_invoke.assert_called_once()