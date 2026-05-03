import os
import base64
import hashlib
import re
from io import BytesIO
from typing import List, Dict, Any, Optional

# OS Dependencies Check
try:
    from pdf2image import convert_from_path
    from PIL import Image
except ImportError:
    print("❌ ERROR: Missing dependencies. Run: pip install pdf2image pillow")
    raise

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma


class CVVectorManager:
    """
    Manages CV ingestion and retrieval with structure-aware chunking.
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
        self.embeddings = embeddings  # type: ignore
        self._vectorstore: Optional[Chroma] = None
        self.hash_file_path = os.path.join(self.db_path, "cv_hash.txt")
        os.makedirs(self.db_path, exist_ok=True)

    def _init_vectorstore(self) -> Chroma:
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.db_path,
        )

    def _ensure_vectorstore_ready(self):
        if self._vectorstore is None:
            self._vectorstore = self._init_vectorstore()

        if os.path.exists(self.hash_file_path):
            collection_data: Dict[str, Any] = self._vectorstore.get()
            if not collection_data or not collection_data.get("documents"):
                raise RuntimeError("Vector DB is empty or corrupted.")
        else:
            raise RuntimeError("CV not ingested yet.")

    @staticmethod
    def _calculate_file_hash(file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    @staticmethod
    def _pdf_to_base64_images(file_path: str) -> List[str]:
        images = convert_from_path(file_path, dpi=300)
        base64_images = []

        for img in images:
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_images.append(img_str)

        return base64_images

    @staticmethod
    def _normalize_bullets(text: str) -> str:
        return re.sub(r"[•\-\*]", "\n•", text)

    @staticmethod
    def _split_experience_block(text: str) -> List[str]:
        """
        Splits experience section into job-level chunks using ### headers.
        """
        jobs = re.split(r"\n(?=### )", text)
        return [job.strip() for job in jobs if job.strip()]

    def ingest_cv(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Resume not found at: {file_path}")

        new_hash = self._calculate_file_hash(file_path)
        stored_hash = None

        if os.path.exists(self.hash_file_path):
            with open(self.hash_file_path, "r") as f:
                stored_hash = f.read().strip()

        if new_hash == stored_hash and os.path.exists(self.db_path):
            print("✅ CV unchanged. Using cached embeddings.")
            self._vectorstore = self._init_vectorstore()
            return

        print("👁️ Reading CV via Vision model...")

        base64_images = self._pdf_to_base64_images(file_path)

        clean_text_parts = []

        for i, b64_img in enumerate(base64_images):
            print(f"Processing page {i + 1}/{len(base64_images)}")

            system_msg = SystemMessage(
                content="You are an expert recruitment assistant specializing in CV transcription."
            )

            human_msg = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Transcribe the CV page in Markdown.\n"
                            "# for name\n"
                            "## for sections (Experience, Education, Skills)\n"
                            "### for entries (jobs, degrees)"
                        ),
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

        full_text = "\n\n".join(clean_text_parts)
        full_text = self._normalize_bullets(full_text)

        print(f"✅ Text reconstructed ({len(full_text)} chars)")

        # --- Markdown Header Split ---
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )

        md_docs = markdown_splitter.split_text(full_text)

        # --- Improved Recursive Splitter ---
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=50,
            separators=["\n\n", "\n", "•", ". "],
        )

        final_chunks: List[Document] = []

        for doc in md_docs:
            section = (doc.metadata.get("Header 2") or "").lower()

            # --- EXPERIENCE: custom job-level splitting ---
            if "experience" in section:
                jobs = self._split_experience_block(doc.page_content)

                for job in jobs:
                    if len(job) > 900:
                        sub_chunks = text_splitter.split_text(job)
                        for chunk in sub_chunks:
                            final_chunks.append(
                                Document(
                                    page_content=chunk,
                                    metadata={**doc.metadata, "section": "experience"},
                                )
                            )
                    else:
                        final_chunks.append(
                            Document(
                                page_content=job,
                                metadata={**doc.metadata, "section": "experience"},
                            )
                        )

            # --- OTHER SECTIONS ---
            else:
                chunks = text_splitter.split_text(doc.page_content)

                for chunk in chunks:
                    final_chunks.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                **doc.metadata,
                                "section": section or "general",
                            },
                        )
                    )

        if not final_chunks:
            raise RuntimeError("No chunks generated from CV.")

        # --- Store ---
        self._vectorstore = Chroma.from_documents(
            documents=final_chunks,
            embedding=self.embeddings,
            persist_directory=self.db_path,
            collection_name=self.collection_name,
            ids=[f"id_{i}" for i in range(len(final_chunks))],
        )

        with open(self.hash_file_path, "w") as f:
            f.write(new_hash)

        print(f"✅ Stored {len(final_chunks)} structured chunks.")

    def get_context(self, query: str, k: int = 4) -> str:
        self._ensure_vectorstore_ready()
        docs = self._vectorstore.similarity_search(query, k=k)

        context_parts = []

        for doc in docs:
            headers = [v for k, v in doc.metadata.items() if k.startswith("Header")]
            section_path = " > ".join(headers)
            prefix = f"[Section: {section_path}]\n" if section_path else ""
            context_parts.append(f"{prefix}{doc.page_content}")

        return "\n---\n".join(context_parts)

    def get_full_resume_text(self) -> str:
        self._ensure_vectorstore_ready()
        all_docs = self._vectorstore.get()

        if not all_docs or not all_docs.get("documents"):
            raise RuntimeError("No documents found.")

        return "\n".join(all_docs["documents"])