from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.tools.vectordb import CVVectorManager
from src import utils
from dotenv import load_dotenv
from agents.agents import VISION_MODEL_NAME, EMBEDDED_MODEL_NAME

load_dotenv()

DB_PATH = "../../data/chroma_db"


def inspect_db():
    common_params = {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": utils.get_api_key("OPENROUTER_API_KEY"),
    }

    vision_model = ChatOpenAI(model=VISION_MODEL_NAME, temperature=0, **common_params)

    embeddings = OpenAIEmbeddings(model=EMBEDDED_MODEL_NAME, **common_params)

    manager = CVVectorManager(
        vision_model=vision_model, embeddings=embeddings, db_path=DB_PATH
    )

    # 1. Get all raw data
    raw_data = manager._init_vectorstore().get()
    ids = raw_data["ids"]
    documents = raw_data["documents"]
    metadatas = raw_data["metadatas"]

    print(f"--- DATABASE INSPECTION ---")
    print(f"Total chunks in DB: {len(ids)}")
    print("-" * 30)

    # 2. Show the first 5 chunks to see how they are split
    for i in range(min(5, len(documents))):
        print(f"CHUNK #{i + 1} (ID: {ids[i]})")
        # print(f"Metadata: {metadatas[i]}")
        print(f"Content Preview: {documents[i][:300]}...")
        print("-" * 15)

    # 3. Test a manual query
    test_query = "Python Developer AI ML"
    print(f"\nTesting RAG retrieval for: '{test_query}'")
    context = manager.get_context(test_query, k=2)
    print(f"Retrieved Context:\n{context}")


if __name__ == "__main__":
    inspect_db()
