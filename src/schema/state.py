from typing import Annotated, List, TypedDict, Optional
import operator
from pydantic import BaseModel, Field


class JobOffer(BaseModel):
    """
    Structured model for a single job posting.
    Using Pydantic allows for easy validation and LLM structured output.
    """

    id: str = Field(description="Unique identifier for the job post")
    title: str = Field(description="Job title")
    company: str = Field(description="Company name")
    description: str = Field(description="Full text of the job description")
    url: str = Field(description="Direct link to the posting")
    salary_range: Optional[str] = Field(
        default="N/A", description="Salary info if available"
    )
    match_score: float = Field(
        default=0.0, description="Semantic match score from 0 to 1"
    )
    analysis: Optional[str] = Field(
        default=None, description="Orchestrator's reasoning for this match"
    )


class AgenticHireState(TypedDict, total=False):
    """
    The shared state of the LangGraph workflow.
    """

    # The raw text or summarized version of your CV retrieved from the Vector DB
    resume_context: str

    # Target criteria for job search
    target_criteria: str

    # List of all raw job postings found by the Scout
    # Annotated with operator.add means new results are appended to the list
    found_jobs: Annotated[List[JobOffer], operator.add]

    # List of jobs that have been validated
    valid_jobs: List[JobOffer]

    # Jobs that passed the Orchestrator's quality gate
    shortlisted_jobs: List[JobOffer]

    # Final generated documents (Cover letters, CV tweaks)
    # Key: job_id, Value: dict with content
    applications: dict

    # A list of technical keywords or requirements extracted from the CV
    # to guide the Scout's search
    search_queries: List[str]

    # Track which step we are currently in or any errors
    status: str

    # Target number of valid offers to find
    max_offers: int

    # Track how many times the scout has run to prevent infinite loops
    scout_runs: int
