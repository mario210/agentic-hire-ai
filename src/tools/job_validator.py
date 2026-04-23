import requests
from bs4 import BeautifulSoup
from typing import List, Tuple
from src.schema.state import JobOffer

class JobValidator:
    """
    Validates job postings to ensure they are accessible and currently active.
    """
    
    @staticmethod
    def is_job_valid(job: JobOffer) -> Tuple[bool, str]:
        if not job.url or not job.url.startswith("http"):
            return False, "Invalid URL format"
            
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            # Perform a GET request to analyze the page content
            response = requests.get(job.url, headers=headers, timeout=10)
            
            # 1. Check for dead links (404, 500, etc.)
            if response.status_code >= 400:
                return False, f"HTTP Error {response.status_code}"
                
            # 2. Check page content for signs of expiration
            soup = BeautifulSoup(response.text, "html.parser")
            text_content = soup.get_text(separator=" ", strip=True).lower()
            
            expired_phrases = [
                "no longer accepting applications",
                "offer expired",
                "job has expired",
                "this position has been filled",
                "posting is closed",
                "job is no longer available",
                "this job is closed",
                "role has been filled",
                "404 not found",
                "page not found",
                "not accepting applications",
                "position closed",
                "sorry, we cannot display this page"
            ]
            
            for phrase in expired_phrases:
                if phrase in text_content:
                    return False, f"Found expired phrase: '{phrase}'"
                    
            return True, ""
            
        except requests.exceptions.RequestException as e:
            # If the request fails entirely (timeout, DNS error), it's invalid
            return False, f"Request failed: {str(e)}"

    @staticmethod
    def filter_valid_jobs(jobs: List[JobOffer]) -> List[JobOffer]:
        print("🔍 Validating found job URLs to ensure they are active and not expired...")
        valid_jobs = []
        for job in jobs:
            is_valid, reason = JobValidator.is_job_valid(job)
            if is_valid:
                valid_jobs.append(job)
            else:
                title = getattr(job, 'title', 'Unknown Title')
                print(f"   ❌ Dropping dead or expired job link: [{job.url}] {title} - {reason}")
        return valid_jobs
