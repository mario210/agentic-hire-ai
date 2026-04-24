import requests
from bs4 import BeautifulSoup
from src.schema.state import JobOffer
from pydantic import BaseModel, Field
from loguru import logger


class ExpirationCheck(BaseModel):
    is_active: bool = Field(
        description="True if the job posting is currently active and accepting applications. False if it is expired, closed, not found, or filled."
    )
    reason: str = Field(
        description="A short explanation of why the job is considered active or expired."
    )


class JobValidator:
    """
    Validates job postings to ensure they are accessible and currently active.
    """

    def __init__(self, llm):
        """
        Initialize the JobValidator with an LLM instance.
        """
        self.checker = llm.with_structured_output(ExpirationCheck)

    def is_job_valid(self, job: JobOffer) -> bool:
        if not job.url or job.url == "N/A" or not job.url.startswith("http"):
            logger.warning(f"Invalid URL format for job '{job.title}': {job.url}")
            return False

        try:
            logger.info(f"Validating job '{job.title}' at URL: {job.url}")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            # Perform a GET request to analyze the page content
            response = requests.get(job.url, headers=headers, timeout=10)

            # 1. Check for dead links (404, 500, etc.)
            if response.status_code >= 400:
                logger.warning(
                    f"HTTP Error {response.status_code} when accessing {job.url}"
                )
                return False

            # 2. Check page content for signs of expiration using an LLM
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove scripts and styles to extract clean text
            for script in soup(["script", "style"]):
                script.decompose()

            text_content = soup.get_text(separator=" ", strip=True)

            # We take the first 6000 characters which usually contain expiration banners or 404 messages
            text_to_analyze = text_content[:6000]

            prompt = f"""
            Analyze the following text extracted from a job posting webpage.
            Determine if the job posting is still active or if it has expired, closed, or the position has been filled.
            Pay attention to phrases indicating the job is no longer available in ANY language.
            
            Webpage Text:
            {text_to_analyze}
            
            
            """

            logger.debug(f"Requesting LLM expiration check for job '{job.title}'")
            result = self.checker.invoke(prompt)

            if not result.is_active:
                logger.info(
                    f"Job '{job.title}' is expired/inactive. Reason: {result.reason}"
                )
                return False

            logger.info(f"Job '{job.title}' is active.")
            return True

        except requests.exceptions.RequestException as e:
            # If the request fails entirely (timeout, DNS error), it's invalid
            logger.error(f"Request failed for {job.url}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Validation error for {job.url}: {str(e)}")
            return False
