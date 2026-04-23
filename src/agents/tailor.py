from langchain_core.messages import SystemMessage, HumanMessage
from src.schema.state import AgenticHireState
from urllib.parse import urlparse


class TailorAgent:
    """
    The Tailor takes the shortlisted jobs and the candidate's CV context
    to generate highly personalized application materials.
    """

    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state: AgenticHireState) -> dict:
        print("--- [NODE] EXECUTING TAILOR (CONTENT GENERATION) ---")

        shortlisted_jobs = state.get("shortlisted_jobs", [])
        resume_context = state.get("resume_context", "")

        if not shortlisted_jobs:
            print("⚠️ No shortlisted jobs found. Tailor has nothing to do.")
            return {"status": "Tailor skipped: No jobs to process."}

        applications = {}

        for job in shortlisted_jobs:
            print(f"Generating application materials for: {job.title}...")

            prompt = f"""
            You are a highly critical and skeptical Career Advisor. 
            Your goal is to evaluate if it's genuinely worth applying for this job, returning ONLY a single sentence.

            CANDIDATE CV CONTEXT:
            {resume_context}

            TARGET JOB:
            Title: {job.title}
            Company: {job.company}
            Description: {job.description}

            MATCH REASONING (from Orchestrator):
            {job.analysis}

            INSTRUCTIONS:
            1. Analyze the match between the CV and the job description.
            2. Be skeptical. Look for reasons why it might NOT be a great fit (e.g., missing skills, seniority mismatch).
            3. Write EXACTLY ONE concise sentence stating whether it's worth applying or not, and briefly why.
            """

            # Generate the content
            response = self.llm.invoke([
                SystemMessage(content="You are a highly critical and skeptical Career Advisor."),
                HumanMessage(content=prompt)
            ])

            # Extract portal from URL
            portal = "Unknown Portal"
            if job.url:
                try:
                    portal = urlparse(job.url).netloc.replace("www.", "")
                except Exception:
                    pass

            founded_job_offer = f"{portal} -> {job.url}\n\n{response.content}"

            applications[job.id] = {
                "founded_job_offer": founded_job_offer,
                "job_title": job.title,
                "company": job.company
            }

        return {
            "applications": applications,
            "status": f"Tailor generated {len(applications)} personalized applications."
        }
