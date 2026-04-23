import os
import base64
from io import BytesIO
from typing import List

# OS Dependencies Check
try:
    from pdf2image import convert_from_path
    from PIL import Image
except ImportError:
    print("❌ ERROR: Missing dependencies. Run: pip install pdf2image pillow")
    raise

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from src import utils


class CVVectorManager:
    """
    Manages the lifecycle of CV data within the Vector Database.
    Uses GPT-4o Vision to 'read' the CV as images for perfect text extraction,
    bypassing corrupted PDF structures.
    """

    def __init__(self, db_path: str = "data/chroma_db", collection_name: str = "cv_collection",
                 vision_model_name = "openai/gpt-4o",
                 embedded_model="text-embedding-3-small"):
        self.db_path = db_path
        self.collection_name = collection_name

        # 1. Vision Model (e.g., GPT-4o from OpenAI directly for best vision)
        # Assuming utils.get_api_key returns a valid key
        self.api_key = utils.get_api_key("OPENROUTER_API_KEY")
        self.vision_model = ChatOpenAI(
            model=vision_model_name,
            temperature=0,
            base_url="https://openrouter.ai/api/v1",
            api_key=utils.get_api_key("OPENROUTER_API_KEY"),
        )

        # 2. Embeddings Model (Using your OpenRouter config)
        self.embeddings = OpenAIEmbeddings(
            model=embedded_model,
            base_url="https://openrouter.ai/api/v1",
            api_key=utils.get_api_key("OPENROUTER_API_KEY"),
        )
        self._vectorstore = None

    def _init_vectorstore(self):
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.db_path
        )

    def _pdf_to_base64_images(self, file_path: str) -> List[str]:
        """
        Converts PDF pages into base64 encoded strings for the LLM.
        """
        try:
            images = convert_from_path(file_path, dpi=300)
            base64_images = []

            for img in images:
                buffered = BytesIO()
                # Convert to JPEG to reduce token usage vs PNG
                img.save(buffered, format="JPEG", quality=85)
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                base64_images.append(img_str)

            return base64_images
        except Exception as e:
            print(f"❌ Error converting PDF to images: {e}")
            raise

    def ingest_cv(self, file_path: str):
        """
        The multimodal ingestion pipeline:
        PDF -> Images -> Vision LLM -> Clean Text -> Chunks -> ChromaDB.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Resume not found at: {file_path}")

        print(f"👁️ AgenticHire Vision is 'reading' {file_path}. This may take a minute...")

        # 1. Convert PDF to Images
        base64_images = self._pdf_to_base64_images(file_path)

        # 2. Multimodal Transcription
        clean_text_parts = []
        for i, b64_img in enumerate(base64_images):
            print(f"  Processing page {i + 1}/{len(base64_images)}...")

            system_msg = SystemMessage(
                content="You are an expert recruitment assistant specializing in CV transcription.")

            # We construct a multimodal message
            human_msg = HumanMessage(content=[
                {
                    "type": "text",
                    "text": (
                        "Transcribe the following CV page precisely. Retain the "
                        "original layout, bullet points, and structure. "
                        "Crucially, capture all technical skills, dates, job titles, "
                        "and project descriptions."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_img}",
                        "detail": "high"  # Ensures high-resolution processing
                    }
                }
            ])

            response = self.vision_model.invoke([system_msg, human_msg])
            clean_text_parts.append(response.content)

        full_reconstructed_text = "\n\n".join(clean_text_parts)
        print(f"✅ Reconstructed CV text (length: {len(full_reconstructed_text)})")

        # 3. Text Chunking (Standard RAG from here)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,  # Increased context per chunk
            chunk_overlap=200,
            separators=["\n\n", "\n", "•", ". ", " "]  # Multi-level separators
        )

        chunks = text_splitter.split_text(full_reconstructed_text)

        # 4. Create Vector Store
        self._vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_path,
            collection_name=self.collection_name,
            ids=[f"id_{i}" for i in range(len(chunks))]  # Optional but good practice
        )

        print(f"✅ Vector DB updated. Successfully stored {len(chunks)} clean CV chunks.")

    def get_context(self, query: str, k: int = 4) -> str:
        if self._vectorstore is None: self._vectorstore = self._init_vectorstore()
        docs = self._vectorstore.similarity_search(query, k=k)
        return "\n---\n".join([doc.page_content for doc in docs])

    def get_full_resume_text(self) -> str:
        if self._vectorstore is None:
            self._vectorstore = self._init_vectorstore()

        # Retrieve all documents using the newer API
        all_docs = self._vectorstore.get()
        return "\n".join(all_docs['documents'])