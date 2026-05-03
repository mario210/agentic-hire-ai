from datetime import datetime
from typing import List
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
)
from src.schema.state import AgenticHireState, JobOffer
from src.tools.search import job_search_tool
from src.tools.scrape import scrape_webpage_tool
from src.utils import JobParser
from loguru import logger


class ScoutAgent:
    """
    The ScoutAgent analyzes the candidate's CV and uses OrioSearch
    to find relevant job postings. It also scrapes the found portals to extract
    concrete job offers instead of just search pages.
    """

    def __init__(self, llm):
        # Bind both search and scrape tools to the LLM
        self.llm = llm.bind_tools([job_search_tool, scrape_webpage_tool])
        self.parser = JobParser()

    def __call__(self, state: AgenticHireState) -> dict:
        scout_runs = state.get("scout_runs", 0) + 1
        logger.info(f"--- [NODE] EXECUTING SCOUT AGENT (Run {scout_runs}) ---")

        resume_context = state.get("resume_context", "No resume context provided.")
        # target_criteria is not in the type definition, fallback correctly
        target_criteria = state.get("target_criteria") or "open job roles matching the candidate's CV"

        # Extract previously evaluated jobs to avoid duplicates on subsequent runs
        evaluated_jobs = state.get("found_jobs", [])
        rejected_jobs = state.get("rejected_jobs", [])

        all_prior_jobs = evaluated_jobs + rejected_jobs
        titles_to_avoid = [
            job.title for job in all_prior_jobs if hasattr(job, "title") and job.title
        ]
        rejected_urls = {job.url.rstrip('/') for job in rejected_jobs if hasattr(job, "url") and job.url}
        existing_urls = {job.url.rstrip('/') for job in evaluated_jobs if hasattr(job, "url") and job.url}
        urls_to_avoid = existing_urls | rejected_urls

        logger.debug(f"Previously evaluated jobs count: {len(titles_to_avoid)}")

        # Add a slight variation to the prompt on subsequent runs to encourage new results
        search_variation = ""
        if scout_runs > 1:
            search_variation = f" This is search attempt #{scout_runs}. Try finding different, more recent, or alternative job postings than before."
            if titles_to_avoid or urls_to_avoid:
                search_variation += f"\nIMPORTANT: Skip these previously evaluated/rejected jobs: {', '.join(titles_to_avoid)}. Also avoid any jobs from these URLs: {', '.join(urls_to_avoid)}"
                logger.debug(
                    "Added search variation to avoid previously evaluated jobs."
                )

        current_date = datetime.now().strftime("%Y-%m-%d")

        system_msg = SystemMessage(
            content=(
                "You are a professional Recruitment Scout. Your task is to find CONCRETE, ACTIVE job offers, not just search portal pages.\n"
                f"Today's date is {current_date}. Use this to determine if a job posting is old or expired.\n"
                "PRIORITY RULES:\n"
                "1. The “target_criteria” is the PRIMARY source of truth and MUST be strictly followed.\n"
                "2. The CV is SECONDARY and should be used only to refine relevance (skills, experience level, technologies).\n"
                "3. If there is any conflict between the CV and target_criteria, ALWAYS follow the target_criteria.\n"
                "STEPS:\n"
                "Step 1: Use the 'job_search_tool' to find job portals or specific job openings that match the candidate's CV. IMPORTANT: Do NOT restrict your search queries using 'site:' operators (e.g., site:linkedin.com). Search the broader web to find diverse opportunities across all company career pages and job boards.\n"
                "Step 2: If the search returns a job portal or a list page, use the 'scrape_webpage_tool' to open that URL and find concrete job postings.\n"
                "Step 3: IMPORTANT: Check the scraped content of each job offer for signs that it is expired, closed, or no longer accepting applications (e.g., 'offer expired', 'job is closed', 'position filled'). If it is expired, discard it and search for another one.\n"
                "Step 4: Once you have identified valid, active jobs, write a comprehensive final message containing the exact Title, Company, FULL Description, and concrete URL for ONLY the approved active jobs. Do not mention or include discarded jobs in this final summary."
                f"{search_variation}"
            )
        )

        human_msg = HumanMessage(
            content=(
                f"Candidate CV:\n{resume_context}\n\n" f"Preferences: {target_criteria}"
            )
        )

        messages: List[BaseMessage] = [system_msg, human_msg]

        all_found_jobs: List[JobOffer] = []

        logger.info("[SCOUT] Starting LLM interaction loop for searching and scraping.")
        for i in range(3):  # max 3 iterations
            logger.debug(f"LLM interaction loop iteration {i + 1}")
            response = self.llm.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                logger.debug("No tool calls made by the LLM. Exiting loop.")
                break

            for tool_call in response.tool_calls:
                logger.debug(f"[SCOUT] Executing tool: {tool_call['name']}")
                if tool_call["name"] == "job_search_tool":
                    raw_results = job_search_tool.invoke(tool_call["args"])
                    messages.append(
                        ToolMessage(
                            name="job_search_tool",
                            tool_call_id=tool_call["id"],
                            content=str(raw_results),
                        )
                    )
                elif tool_call["name"] == "scrape_webpage_tool":
                    raw_results = scrape_webpage_tool.invoke(tool_call["args"])
                    messages.append(
                        ToolMessage(
                            name="scrape_webpage_tool",
                            tool_call_id=tool_call["id"],
                            content=str(raw_results),
                        )
                    )

        # Ensure we have a final AI summary if the loop maxed out on tool calls
        if messages and getattr(messages[-1], "type", "") == "tool":
            logger.debug("Forcing final LLM summarization after tool executions.")
            final_response = self.llm.invoke(messages)
            messages.append(final_response)

        logger.info("Parsing found jobs from LLM messages.")
        # Only parse the AI's evaluated summaries to prevent parsing raw candidate CVs or discarded/expired scraped text
        raw_text_to_parse = ""
        for msg in messages:
            if getattr(msg, "type", "") == "ai" and not getattr(msg, "tool_calls", []):
                if hasattr(msg, "content") and msg.content:
                    raw_text_to_parse += msg.content + "\n\n"

        if raw_text_to_parse:
            all_found_jobs = self.parser.parse(raw_text_to_parse)
            # Filter out duplicates (already in found_jobs) and explicitly rejected jobs
            all_found_jobs = [
                job for job in all_found_jobs
                if not (hasattr(job, "url") and job.url and job.url.rstrip('/') in urls_to_avoid)
            ]
            logger.debug(f"Parsed {len(all_found_jobs)} jobs from messages.")

        # Fallback if no tool calls were made or no jobs found
        if not all_found_jobs:
            logger.warning(
                "[SCOUT] No jobs found through agent loop. Running fallback search..."
            )
            fallback_query = (
                f"{target_criteria} active jobs recently posted"
            )
            raw_results = job_search_tool.invoke({"query": fallback_query})
            all_found_jobs = self.parser.parse(str(raw_results))
            # Apply the same filtering to fallback results
            all_found_jobs = [
                job for job in all_found_jobs
                if not (hasattr(job, "url") and job.url and job.url.rstrip('/') in urls_to_avoid)
            ]
            logger.debug(f"Parsed {len(all_found_jobs)} jobs from fallback search.")

        logger.info(f"[SCOUT] Found {len(all_found_jobs)} jobs.")
        return {
            "found_jobs": all_found_jobs,
            "scout_runs": scout_runs,
            "status": f"Scouted {len(all_found_jobs)} opportunities.",
        }
