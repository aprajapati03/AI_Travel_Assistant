"""
Data Ingestion Script for Destination Knowledge Assistant RAG Pipeline.

This script:
1. Recursively reads raw destination travel text and markdown files from the data directory.
2. Extracts source metadata (Source Title, Canonical URL, File Name) from headers or defaults.
3. Splits content into meaningful semantic chunks using RecursiveCharacterTextSplitter.
4. Purges any previous ChromaDB index completely so embeddings are created from scratch.
5. Embeds the chunks using HuggingFace sentence transformers ('all-MiniLM-L6-v2').
6. Persists the new vector index into a local ChromaDB directory.
"""

import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging to monitor pipeline stages and errors clearly
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_metadata(file_path: Path, content: str) -> Tuple[str, str]:
    """
    Extracts source title and canonical URL from document content headers,
    falling back to known mappings or filename-derived defaults.

    Args:
        file_path (Path): Path to the source file.
        content (str): Plaintext/Markdown content of the file.

    Returns:
        Tuple[str, str]: (source_title, url)
    """
    source_title = None
    url = None

    # Search for markdown header tags like '**Source Title**: ...' or 'Source Title: ...'
    m_title = re.search(r"\*{0,2}Source Title\*{0,2}:\s*([^\n\r]+)", content, re.IGNORECASE)
    if m_title:
        source_title = m_title.group(1).strip()

    # Search for URL tags like '**URL**: ...' or 'URL: ...'
    m_url = re.search(r"\*{0,2}URL\*{0,2}:\s*(https?://[^\s\n\r]+)", content, re.IGNORECASE)
    if m_url:
        url = m_url.group(1).strip()

    # Graceful fallback based on filename if not explicitly defined
    if not source_title:
        source_title = file_path.stem.replace("_", " ").title()
    if not url:
        url = "https://www.visitsingapore.com"

    return source_title, url


def load_raw_document(
    file_path: Path,
    source_title: Optional[str] = None,
    url: Optional[str] = None
) -> Document:
    """
    Reads text content from a specified file and constructs a LangChain Document
    injected with source metadata for accurate citation.

    Args:
        file_path (Path): Path object pointing to the text/markdown raw data file.
        source_title (Optional[str]): Descriptive title of the source. If None, extracted from content.
        url (Optional[str]): Source URL. If None, extracted from content.

    Returns:
        Document: A LangChain Document object containing page content and metadata.

    Raises:
        FileNotFoundError: If the input file does not exist at file_path.
        IOError: If reading the file content fails.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at path: {file_path.resolve()}")

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Failed to read file '{file_path}': {e}")
        raise IOError(f"Error reading file '{file_path}': {e}") from e

    # Extract metadata if not explicitly provided
    extracted_title, extracted_url = extract_metadata(file_path, content)
    final_title = source_title or extracted_title
    final_url = url or extracted_url

    metadata: Dict[str, str] = {
        "source_title": final_title,
        "url": final_url,
        "file_name": file_path.name,
    }

    return Document(page_content=content, metadata=metadata)


def chunk_document(
    document: Document,
    chunk_size: int = 700,
    chunk_overlap: int = 100
) -> List[Document]:
    """
    Splits a LangChain Document into smaller semantic chunks using
    RecursiveCharacterTextSplitter while maintaining section boundaries and overlap.

    Separators are prioritized so that Markdown headers (##, ###, ####) and paragraphs
    form natural chunk boundaries, keeping stops, attractions, and advice intact.

    Args:
        document (Document): The input Document to be chunked.
        chunk_size (int): Target character length for each chunk. Default is 700.
        chunk_overlap (int): Number of overlapping characters between adjacent chunks. Default is 100.

    Returns:
        List[Document]: A list of chunked Document objects preserving original metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n## ",     # Major section boundary (Day, Topic, Category)
            "\n### ",    # Subsections (Attraction, Stop, Specific Time)
            "\n#### ",   # Sub-items & details
            "\n\n",      # Paragraph breaks
            "\n",        # Line breaks
            ". ",        # Sentence ends
            " ",         # Words
            ""
        ]
    )

    chunks = splitter.split_documents([document])
    
    # Ensure all chunks strictly carry source metadata
    for chunk in chunks:
        chunk.metadata["source_title"] = document.metadata.get("source_title", "Singapore Travel Guide")
        chunk.metadata["url"] = document.metadata.get("url", "https://www.visitsingapore.com")
        chunk.metadata["file_name"] = document.metadata.get("file_name", "guide.md")

    return chunks


def load_and_chunk_all_documents(
    data_directory: Path,
    chunk_size: int = 700,
    chunk_overlap: int = 100
) -> List[Document]:
    """
    Recursively scans the data directory for all .txt and .md files, loads each
    document, extracts metadata, and splits into semantic chunks.

    Args:
        data_directory (Path): Directory containing travel knowledge files.
        chunk_size (int): Chunk size for text splitter.
        chunk_overlap (int): Chunk overlap for text splitter.

    Returns:
        List[Document]: Complete list of processed chunks across all sources.
    """
    if not data_directory.exists():
        raise FileNotFoundError(f"Data directory not found at path: {data_directory.resolve()}")

    files = sorted(list(data_directory.rglob("*.txt")) + list(data_directory.rglob("*.md")))
    logger.info(f"Discovered {len(files)} source documents in '{data_directory}' (including subdirectories).")

    all_chunks: List[Document] = []
    for file_path in files:
        try:
            doc = load_raw_document(file_path)
            chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            all_chunks.extend(chunks)
            rel_path = file_path.relative_to(data_directory)
            logger.info(
                f"Processed '{rel_path}' -> {len(chunks)} chunks | "
                f"Source: '{doc.metadata['source_title'][:35]}' | URL: {doc.metadata['url']}"
            )
        except Exception as e:
            logger.warning(f"Skipping file '{file_path}' due to error: {e}")

    logger.info(f"Total semantic chunks generated across all sources: {len(all_chunks)}")
    return all_chunks


def initialize_embeddings(
    model_name: str = "all-MiniLM-L6-v2"
) -> HuggingFaceEmbeddings:
    """
    Initializes HuggingFace sentence transformer embeddings model for local vector encoding.

    Args:
        model_name (str): HuggingFace embedding model repository identifier. Default is 'all-MiniLM-L6-v2'.

    Returns:
        HuggingFaceEmbeddings: Initialized HuggingFace embeddings component.
    """
    logger.info(f"Initializing HuggingFace embedding model: '{model_name}'...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        logger.info("HuggingFace embeddings model loaded successfully.")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize HuggingFace embeddings model '{model_name}': {e}")
        raise RuntimeError(f"Embedding initialization error: {e}") from e


def store_chunks_in_chroma(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    persist_directory: str = "./chroma_db",
    collection_name: str = "singapore_knowledge",
    recreate_from_scratch: bool = True
) -> Chroma:
    """
    Embeds document chunks and persists them into a local ChromaDB vector database directory.
    When recreate_from_scratch is True, completely removes any existing ChromaDB directory
    to ensure old embeddings are fully purged.

    Args:
        chunks (List[Document]): Processed document text chunks to store.
        embeddings (HuggingFaceEmbeddings): Embedding model instance.
        persist_directory (str): Local path where ChromaDB database will be stored. Default is './chroma_db'.
        collection_name (str): Vector database collection identifier. Default is 'singapore_knowledge'.
        recreate_from_scratch (bool): If True, deletes existing persist_directory first.

    Returns:
        Chroma: Persisted Chroma vector store object.
    """
    persist_path = Path(persist_directory).resolve()

    if recreate_from_scratch and persist_path.exists():
        logger.info(f"Purging existing ChromaDB directory at '{persist_path}' to rebuild fully from scratch...")
        shutil.rmtree(persist_path, ignore_errors=True)
        logger.info("Old embeddings successfully deleted.")

    logger.info(f"Persisting {len(chunks)} chunks into ChromaDB at '{persist_path}'...")
    try:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(persist_path),
            collection_name=collection_name
        )
        logger.info("ChromaDB vector store successfully created and persisted.")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to create/persist ChromaDB vector store: {e}")
        raise RuntimeError(f"ChromaDB persistence error: {e}") from e


def run_ingestion_pipeline(
    data_directory: Path,
    persist_directory: str = "./chroma_db",
    collection_name: str = "singapore_knowledge",
    chunk_size: int = 700,
    chunk_overlap: int = 100
) -> None:
    """
    Executes the complete end-to-end data ingestion pipeline:
    1. Scan & Load all files in data directory -> 
    2. Extract & Tag metadata (source_title, url) -> 
    3. Meaningful Chunking (RecursiveCharacterTextSplitter) -> 
    4. Purge old ChromaDB directory (Fresh build from scratch) -> 
    5. Generate HuggingFace Vector Embeddings -> 
    6. Persist Chroma Vector Database.

    Args:
        data_directory (Path): Root directory containing travel knowledge documents.
        persist_directory (str): Directory where the vector database will reside.
        collection_name (str): Chroma collection name.
        chunk_size (int): Target character length for each chunk.
        chunk_overlap (int): Overlap characters between chunks.
    """
    try:
        logger.info("=== Starting Data Ingestion Pipeline (From Scratch) ===")

        # Step 1 & 2 & 3: Load, tag metadata, and chunk all documents in data/
        chunks = load_and_chunk_all_documents(
            data_directory=data_directory,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        if not chunks:
            raise ValueError(f"No document chunks were generated from directory: {data_directory}")

        # Step 4: Initialize HuggingFace embeddings
        embeddings = initialize_embeddings(model_name="all-MiniLM-L6-v2")

        # Step 5 & 6: Purge old DB and persist fresh chunks into ChromaDB
        store_chunks_in_chroma(
            chunks=chunks,
            embeddings=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
            recreate_from_scratch=True
        )

        logger.info(f"=== Pipeline Completed: Ingested {len(chunks)} chunks into ChromaDB ===")

    except Exception as e:
        logger.critical(f"Data ingestion pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    chroma_dir = str(base_dir / "chroma_db")

    run_ingestion_pipeline(
        data_directory=data_dir,
        persist_directory=chroma_dir,
        collection_name="singapore_knowledge",
        chunk_size=700,
        chunk_overlap=100
    )
