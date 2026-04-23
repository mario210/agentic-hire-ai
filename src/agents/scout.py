import json
import utils
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from src.schema.state import AgenticHireState, JobOffer
from src.tools.search import job_search_tool  # This is our @tool
from src.utils import JobParser
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

class ScoutAgent:
    """
    The ScoutAgent analyzes the candidate's CV and uses OrioSearch
    to find relevant job postings.
    """

    def __init__(self, model_name: str = "openai/gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            base_url="https://openrouter.ai/api/v1",
            api_key=utils.get_api_key("OPENROUTER_API_KEY"),
        ).bind_tools([job_search_tool])
        self.parser = JobParser()

    def __call__(self, state: AgenticHireState) -> dict:
        print("--- [NODE] EXECUTING SCOUT AGENT ---")

        resume_context = state.get("resume_context", "No resume context provided.")
        target_criteria = state.get("target_criteria", "AI Python Developer roles")

        system_msg = SystemMessage(content=(
            "You are a professional Recruitment Scout. Use the 'job_search_tool' "
            "to find job openings that match the candidate's CV."
        ))

        human_msg = HumanMessage(content=(
            f"Candidate CV:\n{resume_context}\n\n"
            f"Preferences: {target_criteria}"
        ))

        response = self.llm.invoke([system_msg, human_msg])
        all_found_jobs: List[JobOffer] = []

        # 1. Check if the LLM generated tool calls
        if response.tool_calls:
            for tool_call in response.tool_calls:
                # Use .invoke() on the tool object.
                # tool_call['args'] is already a dict: {"query": "..."}
                raw_results = job_search_tool.invoke(tool_call["args"])

                structured_batch = self.parser.parse(str(raw_results))
                all_found_jobs.extend(structured_batch)

        # 2. Fallback if no tool calls were made
        if not all_found_jobs:
            print("⚠️ No tool calls or results. Running fallback search...")
            fallback_query = f"{target_criteria} jobs for someone with: {resume_context[:100]}"
            # We call .invoke with the expected dict schema of the tool
            raw_results = job_search_tool.invoke({"query": fallback_query})
            all_found_jobs = self.parser.parse(str(raw_results))

        return {
            "found_jobs": all_found_jobs,
            "status": f"Scouted {len(all_found_jobs)} opportunities."
        }