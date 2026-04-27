import pytest
import os
import hashlib
import base64
from unittest.mock import MagicMock, patch, ANY
from io import BytesIO

# Adjust import path based on project structure.
# Assuming this test file is in 'tests/' and the source is in 'src/tools/'.
from src.tools.vectordb import CVVectorManager
from langchain_core.documents import Document # Used for mocking similarity_search results

# --- Fixtures ---

@pytest.fixture
def mock_vision_model():
    """Mocks a vision model with a predictable response."""
    mock_model = MagicMock()
    mock_model.invoke.return_value.content = "Transcribed text from image."
    return mock_model

@pytest.fixture
def mock_embeddings():
    """Mocks an embeddings object."""
    mock_embed = MagicMock()
    return mock_embed

@pytest.fixture
def temp_db_path(tmp_path):
    """Provides a temporary directory for the Chroma DB."""
    # Ensure the directory exists for Chroma to persist to, or for os.path.exists checks
    db_dir = tmp_path / "chroma_db"
    db_dir.mkdir(parents=True, exist_ok=True)
    return str(db_dir)

@pytest.fixture
def cv_manager(mock_vision_model, mock_embeddings, temp_db_path):
    """Provides an instance of CVVectorManager."""
    manager = CVVectorManager(
        vision_model=mock_vision_model,
        embeddings=mock_embeddings,
        db_path=temp_db_path,
        collection_name="test_cv_collection",
    )
    return manager

@pytest.fixture
def mock_pdf_file(tmp_path):
    """Creates a dummy PDF file for testing."""
    dummy_pdf_path = tmp_path / "dummy.pdf"
    # Create a minimal dummy PDF content (not a real PDF, just for file existence and hash calculation)
    with open(dummy_pdf_path, "wb") as f:
        f.write(b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 0>>endobj\nxref\n0 3\n0000000000 65535 f\n0000000009 00000 n\n0000000074 00000 n\ntrailer<</Size 3/Root 1 0 R>>startxref\n104\n%%EOF")
    return str(dummy_pdf_path)

@pytest.fixture
def mock_image_conversion():
    """Mocks pdf2image.convert_from_path and PIL.Image.save."""
    with patch('src.tools.vectordb.convert_from_path') as mock_convert_from_path, \
         patch('src.tools.vectordb.Image.Image.save') as mock_image_save:
        # Mock a PIL Image object
        mock_img = MagicMock()
        # Ensure the save method is mocked on the instance, not the class
        mock_img.save = mock_image_save
        # Return a list with one mock image
        mock_convert_from_path.return_value = [mock_img]
        yield mock_convert_from_path, mock_image_save

@pytest.fixture
def mock_chroma():
    """Mocks the Chroma class and its methods."""
    with patch('src.tools.vectordb.Chroma') as mock_chroma_class:
        # Mock the instance returned by Chroma.from_texts and _init_vectorstore
        mock_instance = MagicMock()
        mock_instance.similarity_search.return_value = [
            Document(page_content="Chunk 1"),
            Document(page_content="Chunk 2"),
        ]
        mock_instance.get.return_value = {"documents": ["Full text doc 1", "Full text doc 2"]}

        mock_chroma_class.return_value = mock_instance # For _init_vectorstore
        mock_chroma_class.from_texts.return_value = mock_instance # For ingest_cv

        yield mock_chroma_class, mock_instance

# --- Tests ---

def test_init(cv_manager, temp_db_path):
    """Test that the CVVectorManager initializes correctly."""
    assert cv_manager.db_path == temp_db_path
    assert cv_manager.collection_name == "test_cv_collection"
    assert cv_manager.vision_model is not None
    assert cv_manager.embeddings is not None
    assert cv_manager._vectorstore is None
    assert cv_manager.hash_file_path == os.path.join(temp_db_path, "cv_hash.txt")

def test_calculate_file_hash(tmp_path):
    """Test the static method for calculating file SHA256 hash."""
    file_content = b"test content for hashing"
    test_file = tmp_path / "hash_test.txt"
    with open(test_file, "wb") as f:
        f.write(file_content)

    expected_hash = hashlib.sha256(file_content).hexdigest()
    actual_hash = CVVectorManager._calculate_file_hash(str(test_file))
    assert actual_hash == expected_hash

def test_pdf_to_base64_images_success(mock_image_conversion, mock_pdf_file):
    """Test successful conversion of PDF to base64 images."""
    mock_convert_from_path, mock_image_save = mock_image_conversion
    base64_images = CVVectorManager._pdf_to_base64_images(mock_pdf_file)

    mock_convert_from_path.assert_called_once_with(mock_pdf_file, dpi=300)
    # Check that save was called with a BytesIO object (ANY) and correct format/quality
    mock_image_save.assert_called_once_with(ANY, format="JPEG", quality=85)
    assert len(base64_images) == 1
    assert isinstance(base64_images[0], str)
    # The method returns just the base64 string, not the data URI prefix
    assert "data:image/jpeg;base64," not in base64_images[0]

    # Verify it's a valid base64 string
    try:
        base64.b64decode(base64_images[0])
    except Exception:
        pytest.fail("Returned string is not valid base64")

def test_pdf_to_base64_images_error(mock_image_conversion, mock_pdf_file):
    """Test error handling during PDF to image conversion."""
    mock_convert_from_path, _ = mock_image_conversion
    mock_convert_from_path.side_effect = Exception("PDF conversion failed")

    with pytest.raises(Exception, match="PDF conversion failed"):
        CVVectorManager._pdf_to_base64_images(mock_pdf_file)

def test_ingest_cv_file_not_found(cv_manager):
    """Test that ingest_cv raises FileNotFoundError for non-existent files."""
    with pytest.raises(FileNotFoundError, match="Resume not found at: non_existent.pdf"):
        cv_manager.ingest_cv("non_existent.pdf")

def test_ingest_cv_first_time_success(cv_manager, mock_pdf_file, mock_image_conversion, mock_vision_model, mock_chroma, temp_db_path):
    """Test successful first-time ingestion of a CV."""
    mock_convert_from_path, mock_image_save = mock_image_conversion
    mock_chroma_class, mock_chroma_instance = mock_chroma

    cv_manager.ingest_cv(mock_pdf_file)

    # Assertions for PDF to Image conversion
    mock_convert_from_path.assert_called_once_with(mock_pdf_file, dpi=300)
    mock_image_save.assert_called_once_with(ANY, format="JPEG", quality=85)

    # Assertions for Vision LLM transcription
    mock_vision_model.invoke.assert_called_once()
    args, kwargs = mock_vision_model.invoke.call_args
    human_message_content = args[0][1].content
    assert human_message_content[0]["type"] == "text"
    assert "Transcribe the following CV page precisely..." in human_message_content[0]["text"]
    assert human_message_content[1]["type"] == "image_url"
    assert "data:image/jpeg;base64," in human_message_content[1]["image_url"]["url"]

    # Assertions for ChromaDB creation
    mock_chroma_class.from_texts.assert_called_once()
    _, kwargs = mock_chroma_class.from_texts.call_args # Use _ for unused positional args
    assert "Transcribed text from image." in kwargs["texts"][0] # Check if the transcribed text is passed
    assert kwargs["embedding"] == cv_manager.embeddings
    assert kwargs["persist_directory"] == temp_db_path
    assert kwargs["collection_name"] == "test_cv_collection"
    assert cv_manager._vectorstore == mock_chroma_instance

    # Assert hash file is created and contains the correct hash
    hash_file_path = os.path.join(temp_db_path, "cv_hash.txt")
    assert os.path.exists(hash_file_path)
    with open(hash_file_path, "r") as f:
        stored_hash = f.read().strip()
    assert stored_hash == CVVectorManager._calculate_file_hash(mock_pdf_file)

def test_ingest_cv_cached_unchanged(cv_manager, mock_pdf_file, mock_image_conversion, mock_vision_model, mock_chroma, temp_db_path):
    """Test that ingestion is skipped if the file is unchanged and DB exists."""
    mock_convert_from_path, _ = mock_image_conversion
    mock_chroma_class, mock_chroma_instance = mock_chroma

    # Simulate a previous ingestion by creating the hash file with the current file's hash
    initial_hash = CVVectorManager._calculate_file_hash(mock_pdf_file)
    # The temp_db_path fixture already ensures the directory exists
    with open(cv_manager.hash_file_path, "w") as f:
        f.write(initial_hash)

    # Call ingest_cv
    cv_manager.ingest_cv(mock_pdf_file)

    # Assert that conversion and vision model were NOT called
    mock_convert_from_path.assert_not_called()
    mock_vision_model.invoke.assert_not_called()

    # Assert that Chroma.from_texts was NOT called (no re-ingestion)
    mock_chroma_class.from_texts.assert_not_called()
    # Assert that _init_vectorstore was called to load the existing DB
    mock_chroma_class.assert_called_once_with(
        collection_name="test_cv_collection",
        embedding_function=cv_manager.embeddings,
        persist_directory=temp_db_path,
    )
    assert cv_manager._vectorstore == mock_chroma_instance

def test_ingest_cv_reingest_on_change(cv_manager, mock_pdf_file, mock_image_conversion, mock_vision_model, mock_chroma, temp_db_path):
    """Test that ingestion occurs if the file has changed."""
    mock_convert_from_path, mock_image_save = mock_image_conversion
    mock_chroma_class, mock_chroma_instance = mock_chroma

    # Simulate a previous ingestion with a *different* hash
    # The temp_db_path fixture already ensures the directory exists
    with open(cv_manager.hash_file_path, "w") as f:
        f.write("old_hash_different_from_new_one")

    # Call ingest_cv
    cv_manager.ingest_cv(mock_pdf_file)

    # Assert that conversion and vision model WERE called (re-ingestion occurred)
    mock_convert_from_path.assert_called_once()
    mock_vision_model.invoke.assert_called_once()
    mock_chroma_class.from_texts.assert_called_once()

    # Assert hash file is updated with the new hash
    with open(cv_manager.hash_file_path, "r") as f:
        updated_hash = f.read().strip()
    assert updated_hash == CVVectorManager._calculate_file_hash(mock_pdf_file)
    assert updated_hash != "old_hash_different_from_new_one"

def test_get_context_initial_load(cv_manager, mock_chroma):
    """Test get_context when the vectorstore needs to be initialized."""
    mock_chroma_class, mock_chroma_instance = mock_chroma
    query = "test query"
    k = 2
    context = cv_manager.get_context(query, k=k)

    # _init_vectorstore should be called because _vectorstore is initially None
    mock_chroma_class.assert_called_once_with(
        collection_name="test_cv_collection",
        embedding_function=cv_manager.embeddings,
        persist_directory=cv_manager.db_path,
    )
    mock_chroma_instance.similarity_search.assert_called_once_with(query, k=k)
    assert context == "Chunk 1\n---\nChunk 2"
    assert cv_manager._vectorstore == mock_chroma_instance

def test_get_context_already_loaded(cv_manager, mock_chroma):
    """Test get_context when the vectorstore is already loaded."""
    mock_chroma_class, mock_chroma_instance = mock_chroma
    # Manually set _vectorstore to simulate it being loaded
    cv_manager._vectorstore = mock_chroma_instance
    query = "another query"
    k = 3
    context = cv_manager.get_context(query, k=k)

    # _init_vectorstore should NOT be called again
    mock_chroma_class.assert_not_called()
    mock_chroma_instance.similarity_search.assert_called_once_with(query, k=k)
    assert context == "Chunk 1\n---\nChunk 2"

def test_get_full_resume_text_initial_load(cv_manager, mock_chroma):
    """Test get_full_resume_text when the vectorstore needs to be initialized."""
    mock_chroma_class, mock_chroma_instance = mock_chroma
    full_text = cv_manager.get_full_resume_text()

    # _init_vectorstore should be called because _vectorstore is initially None
    mock_chroma_class.assert_called_once_with(
        collection_name="test_cv_collection",
        embedding_function=cv_manager.embeddings,
        persist_directory=cv_manager.db_path,
    )
    mock_chroma_instance.get.assert_called_once()
    assert full_text == "Full text doc 1\nFull text doc 2"
    assert cv_manager._vectorstore == mock_chroma_instance

def test_get_full_resume_text_already_loaded(cv_manager, mock_chroma):
    """Test get_full_resume_text when the vectorstore is already loaded."""
    mock_chroma_class, mock_chroma_instance = mock_chroma
    # Manually set _vectorstore to simulate it being loaded
    cv_manager._vectorstore = mock_chroma_instance
    full_text = cv_manager.get_full_resume_text()

    # _init_vectorstore should NOT be called again
    mock_chroma_class.assert_not_called()
    mock_chroma_instance.get.assert_called_once()
    assert full_text == "Full text doc 1\nFull text doc 2"