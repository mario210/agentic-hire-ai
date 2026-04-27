from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional

class AppConfig(BaseSettings):
    """
    Application configuration settings.
    Loads environment variables from .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AGENTIC_HIRE_",  # Optional prefix for env vars
        case_sensitive=False,
        extra='ignore',  # Ignore extra env vars not in the model
    )

    # General settings
    debug_mode: bool = Field(True, description="Enable debug mode for more verbose logging and features.")

    # API configuration
    openrouter_base_url: str = Field("https://openrouter.ai/api/v1")
    openrouter_api_key: Optional[str] = None
    oriosearch_base_url: str = Field("http://localhost:8000")

    # AgenticHire AI specific settings
    max_valid_offers: int = Field(3, description="Maximum number of valid job offers to process.")
    max_scout_runs: int = Field(5, description="Maximum number of iterations for the job scout agent.")
    initial_prompt: str = Field(
        "Python Developer or AI Engineer roles. "
        "No Architect, Team Leader or Senior level. "
        "Only consider jobs that are fully remote within Poland or offer hybrid work in Warsaw. "
        "Exclude roles that primarily require Java or non-Python technologies."
    )
    cv_file_path: str = Field("data/cv/sample_cv.pdf", description="Path to the CV file.")

    # LLM settings
    orchestrator_model_name: str = Field("openai/gpt-4o-mini")
    scout_model_name: str = Field("google/gemini-3-flash-preview")
    tailor_model_name: str = Field("openai/gpt-4o-mini")
    vision_model_name: str = Field("openai/gpt-4o")
    validator_model_name: str = Field("openai/gpt-4o")
    embedded_model_name: str = Field("text-embedding-3-small")


config = AppConfig()