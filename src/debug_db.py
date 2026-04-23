from langchain_core._api import path

from src.tools.vectordb import CVVectorManager
from dotenv import load_dotenv

load_dotenv()

def inspect_db():
    path = "../../data/chroma_db"
    manager = CVVectorManager(db_path = path)

    # 1. Get all raw data
    raw_data = manager._init_vectorstore().get()
    ids = raw_data['ids']
    documents = raw_data['documents']
    metadatas = raw_data['metadatas']

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