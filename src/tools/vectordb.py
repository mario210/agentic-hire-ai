import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from src import utils


class CVVectorManager:
    """
    Manages the lifecycle of CV data within the Vector Database.
    Responsible for loading PDFs, chunking text, and semantic retrieval.
    Updated to use langchain-chroma and OpenRouter embeddings.
    """

    def __init__(self, db_path: str = "data/chroma_db", collection_name: str = "cv_collection",
                 embedded_model="text-embedding-3-small"):
        self.db_path = db_path
        self.collection_name = collection_name

        # OpenRouter-specific OpenAIEmbeddings configuration
        self.embeddings = OpenAIEmbeddings(
            model=embedded_model,
            base_url="https://openrouter.ai/api/v1",
            api_key=utils.get_api_key("OPENROUTER_API_KEY"),
        )
        self._vectorstore = None

    def _init_vectorstore(self):
        """
        Private helper to initialize the Chroma client with persistence.
        """
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.db_path
        )

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
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = text_splitter.split_documents(pages)

        # 3. Create and store the vector store
        # In langchain-chroma, this automatically handles persistence
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
        """
        if self._vectorstore is None:
            self._vectorstore = self._init_vectorstore()

        docs = self._vectorstore.similarity_search(query, k=k)
        return "\n---\n".join([doc.page_content for doc in docs])

    def get_full_resume_text(self) -> str:
        """
        Returns the entire CV content.
        """
        if self._vectorstore is None:
            self._vectorstore = self._init_vectorstore()

        # Retrieve all documents using the newer API
        all_docs = self._vectorstore.get()
        return "\n".join(all_docs['documents'])