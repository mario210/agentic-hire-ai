from src.schema.state import AgenticHireState, JobOffer
from src.tools.vectordb import CVVectorManager
from pydantic import BaseModel, Field

class MatchRating(BaseModel):
    """Structured output for the matching logic."""
    score: float = Field(description="Match score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief explanation of why this score was given")


class OrchestratorAgent:
    """
    The Matchmaker. It compares found jobs against the candidate's actual
    experience stored in ChromaDB and filters for the best fits.
    """

    def __init__(self, llm, vector_manager: CVVectorManager):
        self.llm = llm
        # Initialize the Vector DB manager to fetch CV context
        self.vector_manager = vector_manager
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
            {job.title} at {job.company}
            {job.description}

            CANDIDATE EVIDENCE:
            {relevant_cv_parts}

            SCORING RULES:
            - 1.0: Perfect match (all tech stack and seniority levels align).
            - 0.8: Great match (has core skills, maybe missing one secondary skill).
            - 0.6: Good match (has the foundation, can learn the rest).
            - < 0.5: Poor match.

            Consider synonyms (e.g., 'GenAI' matches 'LLM' or 'GPT'). 
            Don't penalize if 'Remote' isn't on the CV if the tech skills are a 100% match.
            """

            rating = self.judge.invoke(prompt)

            # 3. Decision Step: Add to shortlist if it's a strong match
            if rating.score >= 0.65:
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