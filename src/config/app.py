from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class AgenticHireConfig(BaseSettings):
    """
    Configuration settings for the AgenticHire AI system.
    All constants are centralized here for easy management.
    """

    # File paths
    cv_file_path: str = "data/cv/CV.pdf"

    # Job search limits
    max_valid_offers: int = 5
    max_scout_runs: int = 3

    # Initial prompt for job search
    initial_prompt: str = (
        "Junior-level Python Developer or AI Engineer roles. "
        "No Architect, Team Leader or Senior level. "
        "Only consider jobs that are fully remote within Poland or offer hybrid work in Warsaw. "
        "Exclude roles that primarily require Java or non-Python technologies."
    )

    # Agent model names
    orchestrator_model_name: str = "openai/gpt-4o-mini"
    scout_model_name: str = "google/gemini-3-flash-preview"
    tailor_model_name: str = "openai/gpt-4o-mini"
    vision_model_name: str = "openai/gpt-4o"
    validator_model_name: str = "openai/gpt-4o"
    embedded_model_name: str = "text-embedding-3-small"

    # API configuration
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: Optional[str] = None  # Loaded from env

    oriosearch_base_url: str = "http://localhost:8000"

    # Debugging and logging
    debug_mode: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AGENTIC_HIRE_",  # Optional prefix for env vars
        case_sensitive=False,
        extra='ignore',  # Ignore extra env vars not in the model
    )


# Global config instance
config = AgenticHireConfig()
