from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from src.config.settings import config
from pydantic import SecretStr

DB_PATH = "../data/chroma_db"


def inspect_db():
    api_key_value = config.openrouter_api_key
    api_key: SecretStr | None = SecretStr(api_key_value) if api_key_value else None
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model=config.embedded_model_name,
        base_url=config.openrouter_base_url,
        api_key=api_key,
    )

    # Load Chroma DB
    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name="cv_collection",
    )

    # 1. Get all raw data
    raw_data = db._collection.get(include=["documents", "metadatas"])

    ids = raw_data.get("ids", [])
    documents_raw = raw_data.get("documents", [])
    metadatas = raw_data.get("metadatas", [])

    documents = documents_raw if documents_raw is not None else []

    print(f"--- DATABASE INSPECTION ---")
    print(f"Total chunks in DB: {len(ids)}")
    print("-" * 30)

    # 2. Show first 5 chunks
    for i in range(min(5, len(documents))):
        doc_content = documents[i] if isinstance(documents[i], str) else str(documents[i])
        print(f"CHUNK #{i + 1} (ID: {ids[i]})")
        # print(f"Metadata: {metadatas[i]}")
        print(f"Content Preview: {doc_content[:300]}...")
        print("-" * 15)

    # 3. Test similarity search (RAG-style retrieval)
    test_query = "Python Developer AI ML"
    print(f"\nTesting RAG retrieval for: '{test_query}'")

    results = db.similarity_search(test_query, k=2)
    context = "\n\n".join([doc.page_content for doc in results])

    print(f"Retrieved Context:\n{context}")


if __name__ == "__main__":
    inspect_db()
