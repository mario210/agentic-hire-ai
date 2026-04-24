# main.py
from src.graph import app
from src.agents.agents import factory
from loguru import logger
import sys

# Configure loguru
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)

# --- Configuration ---
CV_FILE_PATH = "data/cv/CV.pdf"
MAX_VALID_OFFERS = 5  # The desired number of valid job offers to find
MAX_SCOUT_RUNS = (
    3  # Safeguard: Stop searching after X runs if unable to find good matches
)
INITIAL_PROMPT = "Junior-level Python Developer or AI Engineer roles in Poland. Not Architect or Team Leader or Senior level. Positions should focus on Python, artificial intelligence, or machine learning. Only consider jobs that are fully remote within Poland or offer hybrid work in Warsaw. Exclude roles that primarily require Java or non-Python technologies."

def main():
    logger.info("Starting main process.")

    # 1. Prepare CV (Run once or when CV changes)
    logger.info("Initializing Vector Manager and ingesting CV...")
    cv_manager = factory.vector_manager
    cv_manager.ingest_cv(CV_FILE_PATH)

    # 2. Setup initial state
    # We fetch full text for the initial context
    logger.info("Fetching full resume text for initial context...")
    initial_context = cv_manager.get_full_resume_text()

    initial_state = {
        "resume_context": initial_context,
        "target_criteria": INITIAL_PROMPT,
        "found_jobs": [],
        "shortlisted_jobs": [],
        "applications": {},
        "status": "Starting AgenticHire AI...",
        "max_offers": MAX_VALID_OFFERS,
        "max_scout_runs": MAX_SCOUT_RUNS,
        "scout_runs": 0,
    }

    logger.debug(
        f"Initial state setup with target_criteria: '{initial_state['target_criteria']}' and max_offers: {MAX_VALID_OFFERS}"
    )

    # 3. Run the Graph
    print("🚀 AgenticHire AI is starting...")
    logger.info("Invoking LangGraph application...")
    final_state = app.invoke(initial_state)
    logger.info("LangGraph application finished successfully.")

    # 4. Show Results
    print("\n" + "=" * 30)
    print("🎯 JOB SEARCH SUMMARY")
    print("=" * 30)

    apps = final_state.get("applications", {})
    if not apps:
        print("No applications were generated.")
        logger.warning("No applications were generated in final state.")

    for job_id, content in apps.items():
        print(f"\n📍 {content['job_title']} at {content['company']}")
        if "founded_job_offer" in content:
            print(f"{content['founded_job_offer'][:500]}...")
        print("-" * 20)


if __name__ == "__main__":
    main()
