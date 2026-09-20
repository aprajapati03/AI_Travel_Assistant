"""
Retriever Tool Creation Script for Destination Knowledge Assistant.

This script loads the persisted ChromaDB vector database using the HuggingFace embeddings
model ('all-MiniLM-L6-v2'), converts it into a search retriever, and wraps it into a 
standardized LangChain tool named `search_singapore_knowledge` using `create_retriever_tool`.

Crucially, it injects a custom `document_prompt` to preserve and pass `source_title` and `url`
metadata to the LLM so the agent can cite accurate source links and avoid hallucination.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool, create_retriever_tool
from langchain_huggingface import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_embedding_model(
    model_name: str = "all-MiniLM-L6-v2"
) -> HuggingFaceEmbeddings:
    """
    Initializes and returns the HuggingFace embeddings instance.
    Must match the embedding model used during data ingestion.

    Args:
        model_name (str): HuggingFace embedding model identifier. Default is 'all-MiniLM-L6-v2'.

    Returns:
        HuggingFaceEmbeddings: Initialized embedding model.
    """
    logger.info(f"Loading HuggingFace embedding model: '{model_name}'...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        logger.info("Embedding model loaded successfully.")
        return embeddings
    except Exception as e:
        logger.error(f"Error loading HuggingFace embedding model: {e}")
        raise RuntimeError(f"Could not initialize embeddings: {e}") from e


def load_vector_store(
    persist_directory: str = "./chroma_db",
    collection_name: str = "singapore_knowledge",
    model_name: str = "all-MiniLM-L6-v2"
) -> Chroma:
    """
    Loads an existing persisted Chroma vector database from disk.

    Args:
        persist_directory (str): Path to the persisted ChromaDB directory. Default is './chroma_db'.
        collection_name (str): Collection name used during ingestion. Default is 'singapore_knowledge'.
        model_name (str): Embedding model identifier. Default is 'all-MiniLM-L6-v2'.

    Returns:
        Chroma: Loaded Chroma vector store instance.

    Raises:
        FileNotFoundError: If the persist directory does not exist.
        RuntimeError: If connecting to or loading ChromaDB fails.
    """
    db_path = Path(persist_directory)
    if not db_path.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found at '{db_path.resolve()}'. "
            "Please run 'src/ingest_data.py' first to build and persist the vector index."
        )

    logger.info(f"Connecting to persisted ChromaDB at '{persist_directory}'...")
    try:
        embeddings = load_embedding_model(model_name=model_name)
        vector_store = Chroma(
            persist_directory=str(db_path),
            embedding_function=embeddings,
            collection_name=collection_name
        )
        logger.info("ChromaDB vector store loaded successfully.")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to load Chroma vector store from '{persist_directory}': {e}")
        raise RuntimeError(f"Vector database loading error: {e}") from e


def create_singapore_knowledge_tool(
    persist_directory: str = "./chroma_db",
    collection_name: str = "singapore_knowledge",
    model_name: str = "all-MiniLM-L6-v2",
    k: int = 4
) -> Tool:
    """
    Builds and wraps a LangChain retriever tool for querying Singapore destination knowledge.
    Uses a custom document_prompt to guarantee that Document metadata (Source Title, URL)
    is explicitly returned to the LLM agent for reliable citations.

    Args:
        persist_directory (str): Directory where vector store is persisted.
        collection_name (str): ChromaDB collection name.
        model_name (str): HuggingFace embedding model name.
        k (int): Number of top relevant document chunks to retrieve per search query. Default is 4.

    Returns:
        Tool: A standard LangChain tool object ready to be passed to an AI agent orchestrator.
    """
    # Step 1: Load the persisted Chroma vector store
    vector_store = load_vector_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
        model_name=model_name
    )

    # Step 2: Convert the vector store into a search retriever configured to return top 'k' chunks
    logger.info(f"Converting vector store into retriever with top_k={k}...")
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    # Step 3: Define tool details for LLM Agent selection
    tool_name = "search_singapore_knowledge"
    tool_description = (
        "Use this tool to search local travel knowledge for Singapore, including itineraries, "
        "neighborhoods, attractions, hawker food, culture, transit, and laws. "
        "Input must be a specific natural language search query (e.g., '3-day food itinerary' or 'Chinatown attractions'). "
        "Results contain Source Title, Source URL, and Content. "
        "You MUST cite the specific Source Title and exact Source URL in your final answer whenever you use this information."
    )

    # Step 4: Inject custom document prompt to retain metadata (Source Title & URL)
    document_prompt = PromptTemplate.from_template(
        "Source Title: {source_title}\nSource URL: {url}\nContent:\n{page_content}"
    )

    # Step 5: Wrap the retriever using LangChain's create_retriever_tool with document_prompt
    logger.info(f"Creating retriever tool '{tool_name}' with document_prompt metadata formatting...")
    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name=tool_name,
        description=tool_description,
        document_prompt=document_prompt
    )

    logger.info(f"Retriever tool '{tool_name}' successfully created.")
    return retriever_tool


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    chroma_dir = str(base_dir / "chroma_db")

    try:
        singapore_tool = create_singapore_knowledge_tool(persist_directory=chroma_dir)
        print("\n" + "=" * 60)
        print("RETRIEVER TOOL CREATED SUCCESSFULLY")
        print("=" * 60)
        print(f"Tool Name        : {singapore_tool.name}")
        print(f"Tool Description : {singapore_tool.description}")
        print("=" * 60)

        # Quick verification test query against the retriever
        test_query = "What are some recommended 4-day itineraries in Singapore?"
        print(f"\nExecuting sample verification search for query: '{test_query}'...\n")
        results = singapore_tool.invoke({"query": test_query})
        print("--- RETRIEVAL RESULTS (WITH SOURCE TITLE & URL) ---")
        print(results)
        print("=" * 60 + "\n")

    except FileNotFoundError as fnf_err:
        logger.warning(f"ChromaDB persistence folder not found. Hint: {fnf_err}")
        print("\n[!] Please run 'python src/ingest_data.py' first to build the Chroma database.")
    except Exception as err:
        logger.error(f"Error during retriever tool verification: {err}", exc_info=True)
        sys.exit(1)
