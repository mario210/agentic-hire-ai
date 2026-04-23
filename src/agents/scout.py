from typing import List
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, BaseMessage
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
        target_criteria = state.get("target_criteria", "AI Python Developer roles") if "target_criteria" in state else "AI Python Developer roles"

        # Extract previously evaluated jobs to avoid duplicates on subsequent runs
        evaluated_jobs = state.get("found_jobs", [])
        evaluated_titles = [job.title for job in evaluated_jobs if hasattr(job, 'title')]
        
        logger.debug(f"Previously evaluated jobs count: {len(evaluated_titles)}")

        # Add a slight variation to the prompt on subsequent runs to encourage new results
        search_variation = ""
        if scout_runs > 1:
             search_variation = f" This is search attempt #{scout_runs}. Try finding different, more recent, or alternative job postings than before."
             if evaluated_titles:
                 search_variation += f"\nIMPORTANT: Skip these previously evaluated jobs: {', '.join(evaluated_titles)}"
                 logger.debug("Added search variation to avoid previously evaluated jobs.")

        system_msg = SystemMessage(content=(
            "You are a professional Recruitment Scout. Your task is to find CONCRETE, ACTIVE job offers, not just search portal pages.\n"
            "Step 1: Use the 'job_search_tool' to find job portals or specific job openings that match the candidate's CV.\n"
            "Step 2: If the search returns a job portal or a list page, use the 'scrape_webpage_tool' to open that URL and find concrete job postings.\n"
            "Step 3: IMPORTANT: Check the scraped content of each job offer for signs that it is expired, closed, or no longer accepting applications (e.g., 'offer expired', 'job is closed', 'position filled'). If it is expired, discard it and search for another one.\n"
            "Step 4: Make sure you return information ONLY about specific, active job offers (title, company, description, concrete job URL)."
            f"{search_variation}"
        ))

        human_msg = HumanMessage(content=(
            f"Candidate CV:\n{resume_context}\n\n"
            f"Preferences: {target_criteria}"
        ))

        messages: List[BaseMessage] = [system_msg, human_msg]
        
        all_found_jobs: List[JobOffer] = []
        
        logger.info("Starting LLM interaction loop for searching and scraping.")
        for i in range(3): # max 3 iterations
            logger.debug(f"LLM interaction loop iteration {i + 1}")
            response = self.llm.invoke(messages)
            messages.append(response)
            
            if not response.tool_calls:
                logger.debug("No tool calls made by the LLM. Exiting loop.")
                break
                
            for tool_call in response.tool_calls:
                logger.debug(f"Executing tool: {tool_call['name']}")
                if tool_call["name"] == "job_search_tool":
                    raw_results = job_search_tool.invoke(tool_call["args"])
                    messages.append(
                        ToolMessage(
                            name="job_search_tool",
                            tool_call_id=tool_call["id"],
                            content=str(raw_results)
                        )
                    )
                elif tool_call["name"] == "scrape_webpage_tool":
                    raw_results = scrape_webpage_tool.invoke(tool_call["args"])
                    messages.append(
                        ToolMessage(
                            name="scrape_webpage_tool",
                            tool_call_id=tool_call["id"],
                            content=str(raw_results)
                        )
                    )

        logger.info("Parsing found jobs from LLM messages.")
        # After the loop, the collected tool responses should contain the job details.
        raw_text_to_parse = ""
        for msg in messages:
            if isinstance(msg, ToolMessage):
                raw_text_to_parse += msg.content + "\n\n"
            elif hasattr(msg, 'content') and msg.content:
                raw_text_to_parse += msg.content + "\n\n"

        if raw_text_to_parse:
            all_found_jobs = self.parser.parse(raw_text_to_parse)
            logger.debug(f"Parsed {len(all_found_jobs)} jobs from messages.")

        # Fallback if no tool calls were made or no jobs found
        if not all_found_jobs:
            logger.warning("No jobs found through agent loop. Running fallback search...")
            fallback_query = f"{target_criteria} jobs for someone with: {resume_context[:100]}"
            raw_results = job_search_tool.invoke({"query": fallback_query})
            all_found_jobs = self.parser.parse(str(raw_results))
            logger.debug(f"Parsed {len(all_found_jobs)} jobs from fallback search.")

        logger.info(f"Scout agent finished. Found {len(all_found_jobs)} jobs.")
        return {
            "found_jobs": all_found_jobs,
            "scout_runs": scout_runs,
            "status": f"Scouted {len(all_found_jobs)} opportunities."
        }
