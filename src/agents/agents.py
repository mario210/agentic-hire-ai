from src.agents.scout import ScoutAgent
from src.agents.orchestrator import OrchestratorAgent
from src.agents.tailor import TailorAgent
from src.tools.vectordb import CVVectorManager
from src.tools.job_validator import JobValidator
from src import utils
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

# --- Global Model Configuration ---
SCOUT_MODEL_NAME = "google/gemini-3-flash-preview"
ORCHESTRATOR_MODEL_NAME = "openai/gpt-4o-mini"
TAILOR_MODEL_NAME = "openai/gpt-4o-mini"
VISION_MODEL_NAME = "openai/gpt-4o"
EMBEDDED_MODEL_NAME = "text-embedding-3-small"
VALIDATOR_MODEL_NAME = "openai/gpt-4o-mini"


class AgentFactory:
    """
    Central factory to initialize and provide agents/tools
    with consistent configuration.
    """

    def __init__(self):
        # Centralized OpenRouter Config
        common_params = {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": utils.get_api_key("OPENROUTER_API_KEY"),
        }

        # Initialize the shared components
        vision_model = ChatOpenAI(
            model=VISION_MODEL_NAME, temperature=0, **common_params
        )

        embeddings = OpenAIEmbeddings(model=EMBEDDED_MODEL_NAME, **common_params)

        # Vector manager initialization
        self.vector_manager = CVVectorManager(
            vision_model=vision_model, embeddings=embeddings
        )

        scout_llm = ChatOpenAI(model=SCOUT_MODEL_NAME, temperature=0, **common_params)

        orchestrator_llm = ChatOpenAI(
            model=ORCHESTRATOR_MODEL_NAME, temperature=0, **common_params
        )

        tailor_llm = ChatOpenAI(
            model=TAILOR_MODEL_NAME, temperature=0.7, **common_params
        )

        validator_llm = ChatOpenAI(
            model=VALIDATOR_MODEL_NAME, temperature=0, **common_params
        )

        # Inject them into the agents and tools
        self.scout = ScoutAgent(llm=scout_llm)
        self.orchestrator = OrchestratorAgent(
            llm=orchestrator_llm, vector_manager=self.vector_manager
        )
        self.tailor = TailorAgent(llm=tailor_llm)
        self.job_validator = JobValidator(llm=validator_llm)


# Single instance to be used across the app
factory = AgentFactory()
