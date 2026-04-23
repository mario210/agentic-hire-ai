import utils
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.schema.state import AgenticHireState
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

class TailorAgent:
    """
    The Tailor takes the shortlisted jobs and the candidate's CV context
    to generate highly personalized application materials.
    """

    def __init__(self, model_name: str = "openai/gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.7, # Slightly higher temperature for creativity
            base_url="https://openrouter.ai/api/v1",
            api_key=utils.get_api_key("OPENROUTER_API_KEY"),
        )

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
            You are an expert Career Coach and Technical Writer. 
            Your goal is to write a short, punchy, and professional Cover Letter.

            CANDIDATE CV CONTEXT:
            {resume_context}

            TARGET JOB:
            Title: {job.title}
            Company: {job.company}
            Description: {job.description}

            MATCH REASONING (from Orchestrator):
            {job.analysis}

            INSTRUCTIONS:
            1. Focus on how the candidate's specific experience (CV Context) solves the company's problems.
            2. Keep it under 250 words.
            3. Use a confident but humble tone.
            4. Highlight the match reasoning provided.
            """

            # Generate the content
            response = self.llm.invoke([
                SystemMessage(content="You are a professional technical recruiter."),
                HumanMessage(content=prompt)
            ])

            applications[job.id] = {
                "cover_letter": response.content,
                "job_title": job.title,
                "company": job.company
            }

        return {
            "applications": applications,
            "status": f"Tailor generated {len(applications)} personalized applications."
        }