from src.agents.scout import ScoutAgent
from src.agents.orchestrator import OrchestratorAgent
from src.agents.tailor import TailorAgent
from src.tools.vectordb import CVVectorManager
from src.tools.job_validator import JobValidator
from src.config.settings import config
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr


class AgentFactory:
    """
    Central factory to initialize and provide agents/tools
    with consistent configuration.
    """

    def __init__(self):
        # Centralized OpenRouter Config
        api_key_value = config.openrouter_api_key
        api_key: SecretStr | None = SecretStr(api_key_value) if api_key_value else None

        # Initialize the shared components
        vision_model = ChatOpenAI(
            model=config.vision_model_name,
            temperature=0,
            base_url=config.openrouter_base_url,
            api_key=api_key,
        )

        embeddings = OpenAIEmbeddings(
            model=config.embedded_model_name,
            base_url=config.openrouter_base_url,
            api_key=api_key,
        )

        # Vector manager initialization
        self.vector_manager = CVVectorManager(
            vision_model=vision_model, embeddings=embeddings
        )

        scout_llm = ChatOpenAI(
            model=config.scout_model_name,
            temperature=0,
            base_url=config.openrouter_base_url,
            api_key=api_key,
        )

        orchestrator_llm = ChatOpenAI(
            model=config.orchestrator_model_name,
            temperature=0,
            base_url=config.openrouter_base_url,
            api_key=api_key,
        )

        tailor_llm = ChatOpenAI(
            model=config.tailor_model_name,
            temperature=0.7,
            base_url=config.openrouter_base_url,
            api_key=api_key,
        )

        validator_llm = ChatOpenAI(
            model=config.validator_model_name,
            temperature=0,
            base_url=config.openrouter_base_url,
            api_key=api_key,
        )

        # Inject them into the agents and tools
        self.scout = ScoutAgent(llm=scout_llm)
        self.orchestrator = OrchestratorAgent(
            llm=orchestrator_llm, vector_manager=self.vector_manager
        )
        self.tailor = TailorAgent(llm=tailor_llm)
        self.job_validator = JobValidator(llm=validator_llm)


# Function to get an AgentFactory instance.
# This allows for lazy initialization and easier mocking in tests.
def get_agent_factory():
    """Returns a new instance of AgentFactory."""
    return AgentFactory()
