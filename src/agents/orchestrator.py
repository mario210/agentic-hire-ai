import utils
from langchain_openai import ChatOpenAI
from src.schema.state import AgenticHireState, JobOffer
from src.tools.vectordb import CVVectorManager
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

class MatchRating(BaseModel):
    """Structured output for the matching logic."""
    score: float = Field(description="Match score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief explanation of why this score was given")


class OrchestratorAgent:
    """
    The Matchmaker. It compares found jobs against the candidate's actual
    experience stored in ChromaDB and filters for the best fits.
    """

    def __init__(self, model_name: str = "openai/gpt-4o-mini", embedded_model_name: str = "text-embedding-3-small"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            base_url="https://openrouter.ai/api/v1",
            api_key=utils.get_api_key("OPENROUTER_API_KEY"),
        )
        # Initialize the Vector DB manager to fetch CV context
        self.vector_manager = CVVectorManager(embedded_model=embedded_model_name)
        # Create a structured judge
        self.judge = self.llm.with_structured_output(MatchRating)

    def __call__(self, state: AgenticHireState) -> dict:
        print("--- [NODE] EXECUTING ORCHESTRATOR (MATCHMAKER) ---")

        found_jobs = state.get("found_jobs", [])
        shortlisted_jobs = []

        if not found_jobs:
            print("⚠️ No jobs found to analyze.")
            return {"status": "Orchestrator skipped: No jobs found."}

        for job in found_jobs:
            print(f"Analyzing: {job.title} at {job.company}...")

            # 1. RAG Step: Get specific context from CV for THIS job
            # We search for the job title and description in our vectors
            search_query = f"{job.title} {job.description[:200]}"
            relevant_cv_parts = self.vector_manager.get_context(search_query, k=3)

            # 2. Evaluation Step: Compare Job vs. CV Evidence
            prompt = f"""
            You are an expert Career Matchmaker. Compare the Job Description with the Candidate's Experience.

            JOB DESCRIPTION:
            Title: {job.title}
            Description: {job.description}

            CANDIDATE EVIDENCE (from CV):
            {relevant_cv_parts}

            Rate the match from 0.0 to 1.0. 
            Be strict. Only give > 0.8 if the candidate has the core required technologies.
            """

            rating = self.judge.invoke(prompt)

            # 3. Decision Step: Add to shortlist if it's a strong match
            if rating.score >= 0.75:
                job.match_score = rating.score
                job.analysis = rating.reasoning
                shortlisted_jobs.append(job)
                print(f"✅ Match found! Score: {rating.score}")
            else:
                print(f"❌ Weak match ({rating.score}). Skipping.")

        # Sorting shortlisted jobs by score (descending)
        shortlisted_jobs.sort(key=lambda x: x.match_score, reverse=True)

        return {
            "shortlisted_jobs": shortlisted_jobs,
            "status": f"Orchestrator shortlisted {len(shortlisted_jobs)} jobs."
        }