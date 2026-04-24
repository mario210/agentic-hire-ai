import requests
from config.app import config
from langchain_core.tools import tool
from loguru import logger

@tool
def job_search_tool(query: str) -> str:
    """
    Search the web using OrioSearch API for job postings.
    Input should be a specific search query like 'Senior Python Developer jobs London'.
    Returns a string containing a list of search results with titles, snippets, and URLs.
    """
    logger.debug(f"Search Query: {query}")
    payload = {"query": query, "num_results": 10}

    try:
        response = requests.post(config.oriosearch_base_url, json=payload, timeout=10)
        response.raise_for_status()

        # We return the raw string or JSON-like string for the LLM to parse
        results = response.json()
        return str(results)

    except requests.exceptions.RequestException as e:
        return f"Error connecting to OrioSearch: {str(e)}"


class JobSearchProvider:
    """
    A wrapper class to manage search operations.
    In a real-world scenario, this could handle rotating API keys or
    filtering results before they reach the agent.
    """

    def __init__(self):
        self.search_tool = job_search_tool
