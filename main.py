# main.py
from src.graph import app
from src.tools.vectordb import CVVectorManager
import os
from dotenv import load_dotenv

load_dotenv()

CV_FILE_PATH = "data/cv/CV.pdf"


def main():
    # 1. Prepare CV (Run once or when CV changes)
    cv_manager = CVVectorManager()
    cv_manager.ingest_cv(CV_FILE_PATH)

    # 2. Setup initial state
    # We fetch full text for the initial context
    initial_context = cv_manager.get_full_resume_text()

    initial_state = {
        "resume_context": initial_context,
        "target_criteria": "Python Developer, Remote, AI/ML focus",
        "found_jobs": [],
        "shortlisted_jobs": [],
        "applications": {},
        "status": "Starting AgenticHire AI..."
    }

    # 3. Run the Graph
    print("🚀 AgenticHire AI is starting...")
    final_state = app.invoke(initial_state)

    # 4. Show Results
    print("\n" + "=" * 30)
    print("🎯 JOB SEARCH SUMMARY")
    print("=" * 30)

    apps = final_state.get("applications", {})
    for job_id, content in apps.items():
        print(f"\n📍 {content['job_title']} at {content['company']}")
        print(f"--- COVER LETTER ---\n{content['cover_letter'][:300]}...")
        print("-" * 20)


if __name__ == "__main__":
    main()