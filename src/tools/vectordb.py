import os
import utils
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

class CVVectorManager:
    """
    Manages the lifecycle of CV data within the Vector Database.
    Responsible for loading PDFs, chunking text, and semantic retrieval.
    """

    def __init__(self, db_path: str = "data/chroma_db", collection_name: str = "cv_collection", embedded_model="text-embedding-3-small"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(
            model=embedded_model,
            base_url="https://openrouter.ai/api/v1",
            api_key=utils.get_api_key("OPENROUTER_API_KEY"),
        )

        self._vectorstore = None

    def ingest_cv(self, file_path: str):
        """
        Processes a PDF resume: Loads, splits into chunks, and stores in ChromaDB.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Resume not found at: {file_path}")

        # 1. Load the PDF
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        # 2. Split into chunks
        # We use a small chunk size to keep context specific (e.g., one project per chunk)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = text_splitter.split_documents(pages)

        # 3. Create and persist the vector store
        self._vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_path,
            collection_name=self.collection_name
        )
        print(f"✅ Successfully ingested {len(chunks)} chunks from {file_path}")

    def get_context(self, query: str, k: int = 4) -> str:
        """
        Retrieves the most relevant parts of the CV based on a semantic query.
        Used by the Matchmaker to find proof of skills.
        """
        if not self._vectorstore:
            # Load existing DB if it wasn't just created in this session
            self._vectorstore = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )

        docs = self._vectorstore.similarity_search(query, k=k)
        return "\n---\n".join([doc.page_content for doc in docs])

    def get_full_resume_text(self) -> str:
        """
        Returns the entire CV content. Useful for the Orchestrator to get
        a high-level overview of the candidate.
        """
        if not self._vectorstore:
            self._vectorstore = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )

        # Retrieve all documents stored in the collection
        all_docs = self._vectorstore.get()
        return "\n".join(all_docs['documents'])