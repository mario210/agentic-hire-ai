import os
import base64
import hashlib
from io import BytesIO
from typing import List
from typing import Dict, Any, Optional
# OS Dependencies Check
try:
    from pdf2image import convert_from_path
    from PIL import Image
except ImportError:
    print("❌ ERROR: Missing dependencies. Run: pip install pdf2image pillow")
    raise

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


class CVVectorManager:
    """
    Manages the lifecycle of CV data within the Vector Database.
    Uses GPT-4o Vision to 'read' the CV as images for perfect text extraction,
    bypassing corrupted PDF structures.
    """

    def __init__(
        self,
        vision_model,
        embeddings,
        db_path: str = "data/chroma_db",
        collection_name: str = "cv_collection",
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.vision_model = vision_model
        self.embeddings = embeddings # type: ignore
        self._vectorstore: Optional[Chroma] = None
        self.hash_file_path = os.path.join(self.db_path, "cv_hash.txt")
        os.makedirs(self.db_path, exist_ok=True) # Ensure the persistence directory exists

    def _init_vectorstore(self) -> Chroma:
        """
        Initializes the Chroma client.
        """
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.db_path,
        )

    def _ensure_vectorstore_ready(self):
        """
        Ensures the vector store is initialized and contains documents.
        Raises a RuntimeError if the database is not ready or appears empty
        when a hash file indicates it should be populated.
        """
        if self._vectorstore is None:
            self._vectorstore = self._init_vectorstore()

        # Check if a hash file exists, which implies a CV was successfully ingested.
        # If it exists, we expect the vector store to contain documents.
        if os.path.exists(self.hash_file_path):
            try:
                collection_data: Dict[str, Any] = self._vectorstore.get()
                if not collection_data or not collection_data.get("documents"):
                    raise RuntimeError(
                        f"Vector database at '{self.db_path}' is empty or corrupted, "
                        "but a CV hash file exists. "
                        "Please ensure the CV is correctly ingested. "
                        "You might need to re-run the ingestion process."
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to access or verify vector database at '{self.db_path}'. "
                    f"It might be corrupted or inaccessible. Error: {e}. "
                    "Please check the database path and ensure the CV is ingested."
                )
        else:
            # If no hash file exists, it means ingest_cv was never successfully called.
            # In this case, the vector store should not be used.
            raise RuntimeError(
                "CV has not been ingested into the vector database. "
                "Please call 'ingest_cv()' first to process the resume."
            )

    @staticmethod
    def _calculate_file_hash(file_path: str) -> str:
        """Calculates the SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read and update hash in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    @staticmethod
    def _pdf_to_base64_images(file_path: str) -> List[str]:
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
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                base64_images.append(img_str)

            return base64_images
        except Exception as e:
            print(f"❌ Error converting PDF to images: {e}")
            raise

    def ingest_cv(self, file_path: str):
        """
        The multimodal ingestion pipeline:
        PDF -> Images -> Vision LLM -> Clean Text -> Chunks -> ChromaDB.
        Caches the CV hash to avoid re-processing unchanged files.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Resume not found at: {file_path}")

        # Caching mechanism
        new_hash = self._calculate_file_hash(file_path)
        stored_hash = None
        if os.path.exists(self.hash_file_path):
            with open(self.hash_file_path, "r") as f:
                stored_hash = f.read().strip()

        # If hash is the same and DB exists, skip ingestion
        if new_hash == stored_hash and os.path.exists(self.db_path):
            print(
                f"✅ File '{os.path.basename(file_path)}' is unchanged. Using existing vector data."
            )
            self._vectorstore = self._init_vectorstore()
            return

        print(
            f"👁️ AgenticHire Vision is 'reading' {file_path}. This may take a minute..."
        )

        # 1. Convert PDF to Images
        base64_images = self._pdf_to_base64_images(file_path)

        # 2. Multimodal Transcription
        clean_text_parts = []
        for i, b64_img in enumerate(base64_images):
            print(f"  Processing page {i + 1}/{len(base64_images)}...")

            system_msg = SystemMessage(
                content="You are an expert recruitment assistant specializing in CV transcription."
            )

            human_msg = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Transcribe the following CV page precisely...",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_img}",
                            "detail": "high",
                        },
                    },
                ]
            )

            response = self.vision_model.invoke([system_msg, human_msg])
            clean_text_parts.append(response.content)

        full_reconstructed_text = "\n\n".join(clean_text_parts)
        print(f"✅ Reconstructed CV text (length: {len(full_reconstructed_text)})")

        # 3. Text Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n", "•", ". ", " "],
        )
        chunks = text_splitter.split_text(full_reconstructed_text)

        if not chunks:
            raise RuntimeError(
                f"No text chunks could be generated from CV '{file_path}'. "
                "The CV might be empty or unreadable."
            )

        # 4. Create Vector Store
        self._vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_path,
            collection_name=self.collection_name,
            ids=[f"id_{i}" for i in range(len(chunks))],
        )

        # 5. Save the new hash
        with open(self.hash_file_path, "w") as f:
            f.write(new_hash)

        print(
            f"✅ Vector DB updated. Successfully stored {len(chunks)} clean CV chunks."
        )

    def get_context(self, query: str, k: int = 4) -> str:
        self._ensure_vectorstore_ready()
        docs = self._vectorstore.similarity_search(query, k=k)
        return "\n---\n".join([doc.page_content for doc in docs])

    def get_full_resume_text(self) -> str:
        self._ensure_vectorstore_ready()
        all_docs = self._vectorstore.get()
        if not all_docs or not all_docs.get("documents"):
            # This case should ideally be caught by _ensure_vectorstore_ready, but as a fallback
            raise RuntimeError("No documents found in the vector database after readiness check.")
        return "\n".join(all_docs["documents"])
