## Getting Started

This project uses [uv](https://github.com/astral-sh/uv) for blazing-fast Python dependency and environment management.

### 1. Install `uv`
If you do not have `uv` installed, you can install it globally via pip:
```bash
pip install uv
```
*(Alternatively, use the standalone installer from their official documentation).*

### 2. Sync the Environment
Once cloned, navigate to the project directory and run:
```bash
uv sync
```
This command will automatically create a virtual environment (`.venv`) and install all required dependencies listed in the `pyproject.toml` and `uv.lock` files.

## Project Structure
The project follows a modular architecture designed for scalability and clear separation of concerns between the LangGraph orchestration and the individual AI Agents.
```
agentic-hire-ai/
├── data/                   # Local storage for source files (PDF resumes) and ChromaDB persistent data.
├── src/
│   ├── agents/             # Logic for LangGraph nodes (The "Brains" of the system).
│   │   ├── agents.py       # Core agent definitions and initialization.
│   │   ├── orchestrator.py # Matchmaker logic, decision making, and task planning.
│   │   ├── scout.py        # Job fetching logic (Web Scraping / API integrations).
│   │   └── tailor.py       # Content generation for personalized applications.
│   ├── tools/              # Specialized utilities used by agents to interact with the world.
│   │   ├── job_validator.py# Tool for validating fetched job requirements.
│   │   ├── scrape.py       # Tool for scraping job descriptions from the web.
│   │   ├── search.py       # Search engine API wrappers (e.g., Tavily, Google).
│   │   └── vectordb.py     # Vector DB (RAG) integration for semantic resume matching.
│   ├── schema/             # Data models and shared state definitions.
│   │   └── state.py        # The TypedDict defining the LangGraph State.
│   ├── debug_db.py         # Utility for debugging and inspecting the database.
│   ├── graph.py            # The core LangGraph definition, node connections, and compilation.
│   └── utils.py            # Helper functions (PDF parsing, text formatting, logging).
├── .env                    # Environment variables and API keys (OpenAI, Anthropic, etc.).
└── main.py                 # Main entry point for the CLI application.
```

### Key Architectural Decisions

State Separation: The AgenticHireState is isolated in src/schema/ to prevent circular imports when nodes need to reference the state type.
Decoupled Tools: Agents do not communicate with databases or APIs directly. They use the abstractions in src/tools/, making it easy to swap ChromaDB for Pinecone or change the scraping engine without touching the Agent logic.
Orchestrator-Worker Pattern: orchestrator.py acts as the technical lead, evaluating jobs and delegating "writing" tasks to the tailor.py worker.