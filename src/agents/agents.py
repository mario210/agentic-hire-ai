from src.agents.scout import ScoutAgent
from src.agents.orchestrator import OrchestratorAgent
from src.agents.tailor import TailorAgent
from src.tools.vectordb import CVVectorManager
from src.tools.job_validator import JobValidator
from config.app import config
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class AgentFactory:
    """
    Central factory to initialize and provide agents/tools
    with consistent configuration.
    """

    def __init__(self):
        # Centralized OpenRouter Config
        common_params = {
            "base_url": config.openrouter_base_url,
            "api_key": config.openrouter_api_key,
        }

        # Initialize the shared components
        vision_model = ChatOpenAI(
            model=config.vision_model_name, temperature=0, **common_params
        )

        embeddings = OpenAIEmbeddings(model=config.embedded_model_name, **common_params)

        # Vector manager initialization
        self.vector_manager = CVVectorManager(
            vision_model=vision_model, embeddings=embeddings
        )

        scout_llm = ChatOpenAI(model=config.scout_model_name, temperature=0, **common_params)

        orchestrator_llm = ChatOpenAI(
            model=config.orchestrator_model_name, temperature=0, **common_params
        )

        tailor_llm = ChatOpenAI(
            model=config.tailor_model_name, temperature=0.7, **common_params
        )

        validator_llm = ChatOpenAI(
            model=config.validator_model_name, temperature=0, **common_params
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
