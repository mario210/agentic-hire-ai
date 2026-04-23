import os
import utils
from typing import List
from pydantic import SecretStr, BaseModel
from langchain_openai import ChatOpenAI
from src.schema.state import JobOffer


def get_api_key(key: str):
    """Retrieves the key and securely wraps it."""
    raw_api_key = os.getenv(key)
    return SecretStr(raw_api_key) if raw_api_key else None


class JobOfferList(BaseModel):
    """A collection of job offers extracted from search results."""

    offers: List[JobOffer]


class JobParser:
    """
    Specialized parser that takes raw search snippets and
    converts them into structured JobOffer objects.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        # We use a cheaper/faster model for parsing tasks
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            base_url="https://openrouter.ai/api/v1",
            api_key=utils.get_api_key("OPENROUTER_API_KEY"),
        )

        self.structured_llm = self.llm.with_structured_output(JobOfferList)

    def parse(self, raw_search_results: str) -> List[JobOffer]:
        """
        Processes raw text and returns a list of JobOffer objects.
        """
        system_prompt = """
        You are an expert Data Extraction Agent. 
        Your task is to take raw search engine results and extract structured job posting information.
        If a specific field (like salary) is missing, leave it as 'N/A'.
        Ensure the 'id' is a short, unique string (e.g., 'company-title-hash').
        """

        human_prompt = f"Extract all job postings from these search results:\n\n{raw_search_results}"

        try:
            # The result is automatically an instance of JobOfferList (Pydantic)
            result = self.structured_llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": human_prompt},
                ]
            )
            return result.offers
        except Exception as e:
            print(f"❌ Error during job parsing: {e}")
            return []
