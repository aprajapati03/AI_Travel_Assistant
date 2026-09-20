"""
Streamlit Application for AI Travel Planning Assistant (Phase 3).

This application orchestrates:
1. Local RAG Destination Knowledge Vector Store (ChromaDB + HuggingFace Embeddings)
2. External MCP Tools Server over stdio (Currency Conversion & Weather Forecast)
using LangChain, MultiServerMCPClient, and LangGraph's create_react_agent.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv
import nest_asyncio
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Apply nest_asyncio to allow nested asyncio loops within Streamlit execution context
nest_asyncio.apply()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# LangChain & LangGraph Imports
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# Import local Phase 1 RAG tool creator
from src.retriever_tool import create_singapore_knowledge_tool

# Setup Base Directories
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = str(BASE_DIR / "chroma_db")
MCP_SERVER_SCRIPT = str(BASE_DIR / "mcp_server.py")

# System Prompt enforcing strict agent behavior & routing rules
SYSTEM_PROMPT = """You are an expert AI Travel Assistant specializing in Singapore and global destination planning.
You have access to the following specialized tools:
1. 'search_singapore_knowledge': Searches local vector database for Singapore facts, attractions, hawker food recommendations, itineraries, neighborhoods, transit, culture, and travel etiquette.
2. 'convert_currency': Converts real-time foreign currency amounts via Frankfurter API.
3. 'get_weather': Retrieves live current weather conditions, forecasts for specific dates (e.g. 'YYYY-MM-DD', 'tomorrow', or 'YYYY-MM-DD to YYYY-MM-DD'), or default 3-day forecasts via Open-Meteo API.

STRICT BEHAVIORAL DIRECTIVES:
1. DESTINATION FACTS FROM KNOWLEDGE BASE: For any facts, attractions, hawker food, culture, or travel advice about Singapore, ALWAYS invoke 'search_singapore_knowledge'. Ground your response strictly on retrieved content.
2. SOURCE CITATIONS: Whenever using retrieved destination knowledge, you MUST cite the specific Source Title and canonical URL provided in the retrieved content (e.g., "Source: Visit Singapore | https://...").
3. CURRENT INFORMATION VIA MCP: Use 'get_weather' for real-time weather and forecasts. When the user asks for weather on specific dates (e.g., 'tomorrow', 'on 2026-09-22', or a date range), ALWAYS pass the 'date' parameter (and 'end_date' if a range) to 'get_weather'. Use 'convert_currency' for currency exchanges. Do not use MCP tools for destination knowledge already covered by the local knowledge base.
4. LIVE DATA ATTRIBUTION: When providing information from weather or currency tools, ALWAYS explicitly prefix the statement with: "According to live data..."
5. ANTI-HALLUCINATION & MISSING KNOWLEDGE: Never fabricate facts, exchange rates, weather data, or attraction hours. If a tool fails, is unavailable, or if the knowledge base lacks sufficient information, state clearly that the information is unavailable.
6. DISTINGUISH FACTS FROM SUGGESTIONS: Clearly differentiate between verified factual information (from knowledge base or live tools) and AI-generated suggestions or general travel tips.
7. WEATHER-AWARE ITINERARY PLANNING: For itinerary requests that involve weather forecasts or specific travel dates, check the forecast for those dates using 'get_weather' (providing 'date' and 'end_date' if dates are specified), schedule outdoor attractions on clearer days, and recommend specific indoor alternatives (e.g. Gardens by the Bay domes, ArtScience Museum, National Gallery, Jewel Changi) if rain or thunderstorms are expected.
8. PRESERVE USER PREFERENCES: Maintain and respect user preferences expressed earlier in the conversation (e.g., budget constraints, dietary choices, family with children, interest in culture).
"""


def initialize_session_state() -> None:
    """Initializes Streamlit chat history session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! 👋 I am your AI Travel Planning Assistant. "
                    "I can answer destination questions about Singapore, check live weather forecasts (including specific dates or multi-day trips), "
                    "or convert currencies in real-time. How can I help you plan your trip today?"
                )
            }
        ]


async def load_mcp_tools() -> tuple[MultiServerMCPClient, List[Any]]:
    """
    Asynchronously initializes MultiServerMCPClient and loads tools from the stdio mcp_server.py script.

    Returns:
        tuple[MultiServerMCPClient, List[Any]]: The active MCP client instance and list of retrieved tools.
    """
    logger.info("Initializing MultiServerMCPClient over stdio transport...")
    
    server_config = {
        "travel_tools": {
            "command": sys.executable,  # Uses active virtualenv Python interpreter
            "args": [MCP_SERVER_SCRIPT],
            "transport": "stdio",
        }
    }
    
    try:
        mcp_client = MultiServerMCPClient(server_config)
        mcp_tools = await mcp_client.get_tools()
        logger.info(f"Successfully connected to MCP Server. Loaded {len(mcp_tools)} MCP tools.")
        return mcp_client, mcp_tools
    except Exception as e:
        logger.error(f"Failed to load MCP tools: {e}")
        return None, []


def get_llm_model(api_token: str, model_name: str = "Qwen/Qwen2.5-72B-Instruct") -> Any:
    """
    Instantiates HuggingFaceEndpoint LLM wrapped in ChatHuggingFace.

    Args:
        api_token (str): Hugging Face User Access Token.
        model_name (str): Repository ID of the HuggingFace Instruct model.

    Returns:
        ChatHuggingFace: LangChain Chat Model wrapper around Hugging Face Endpoint.
    """
    logger.info(f"Initializing HuggingFaceEndpoint model '{model_name}'...")
    
    endpoint = HuggingFaceEndpoint(
        repo_id=model_name,
        max_new_tokens=1024,
        do_sample=False,
        huggingfacehub_api_token=api_token,
        timeout=120
    )
    return ChatHuggingFace(llm=endpoint)


@st.cache_resource(show_spinner=False)
def get_cached_rag_tool():
    """Caches the RAG destination knowledge retriever tool in memory to prevent repeated model loading."""
    return create_singapore_knowledge_tool(
        persist_directory=CHROMA_DIR,
        model_name="all-MiniLM-L6-v2"
    )


async def execute_agent_query(
    user_input: str,
    hf_token: str,
    chat_history: List[dict],
    model_name: str = "Qwen/Qwen2.5-72B-Instruct"
) -> tuple[str, List[dict]]:
    """
    Orchestrates RAG and MCP tools using LangGraph's create_react_agent asynchronously.

    Args:
        user_input (str): The latest user prompt.
        hf_token (str): Hugging Face API Token.
        chat_history (List[dict]): Previous session message logs.
        model_name (str): Hugging Face model repository identifier.

    Returns:
        tuple[str, List[dict]]: Final response string and list of tool execution traces.
    """
    # Step 1: Load Local RAG Retriever Tool (Phase 1)
    try:
        rag_tool = get_cached_rag_tool()
        tools = [rag_tool]
    except Exception as e:
        logger.warning(f"Could not load RAG tool: {e}. Proceeding with MCP tools only.")
        tools = []

    # Step 2: Load External MCP Server Tools (Phase 2)
    mcp_client, mcp_tools = await load_mcp_tools()
    if mcp_tools:
        tools.extend(mcp_tools)

    if not tools:
        return "Error: No tools could be loaded. Please ensure vector store and mcp_server.py are available."

    # Step 3: Initialize LLM
    try:
        llm = get_llm_model(api_token=hf_token, model_name=model_name)
    except Exception as e:
        return f"Error initializing LLM: {e}. Please check your Hugging Face API Token.", []

    # Step 4: Construct LangGraph Agent
    logger.info(f"Creating LangGraph Agent with {len(tools)} total tools...")
    agent_executor = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT
    )

    # Step 5: Format Chat History into LangChain Messages
    langchain_messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=msg["content"]))
    
    # Append current user prompt
    langchain_messages.append(HumanMessage(content=user_input))

    # Step 6: Invoke Agent asynchronously
    try:
        inputs = {"messages": langchain_messages}
        result = await agent_executor.ainvoke(inputs)
        
        # Extract execution messages and tool calls
        all_msgs = result.get("messages", [])
        final_text = "No response generated by agent."
        tool_traces = []

        for msg in all_msgs:
            # Check for tool call requests generated by LLM
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_traces.append({
                        "type": "call",
                        "tool": tc.get("name", "tool"),
                        "args": tc.get("args", {})
                    })
            # Check for tool response output
            elif msg.__class__.__name__ == "ToolMessage" or getattr(msg, "type", "") == "tool":
                tool_name = getattr(msg, "name", "tool")
                content = getattr(msg, "content", "")
                tool_traces.append({
                    "type": "result",
                    "tool": tool_name,
                    "content": content
                })
            # Capture final assistant response
            elif isinstance(msg, AIMessage) and msg.content:
                final_text = msg.content

        return final_text, tool_traces
    except Exception as e:
        logger.error(f"Agent execution error: {e}", exc_info=True)
        return f"Error executing agent query: {e}", []


def main() -> None:
    """Main Streamlit User Interface Rendering & Application Logic."""
    st.set_page_config(
        page_title="AI Travel Planning Assistant",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Hide sidebar container and toggle controls completely
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] { display: none !important; }
            [data-testid="collapsedControl"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("✈️ AI Travel Planning Assistant")
    st.caption("Powered by LangChain, LangGraph, Local Chroma RAG & Model Context Protocol (MCP)")

    initialize_session_state()

    # Load configuration from environment variables (.env file)
    load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)
    hf_token = (os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN", "")).strip()
    selected_model = os.getenv("HF_MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct").strip()

    # Render Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("tool_traces"):
                with st.expander("🛠️ Tool Executions & Retrieved Citations", expanded=False):
                    for t in message["tool_traces"]:
                        if t.get("type") == "call":
                            st.markdown(f"**⚡ Tool Invoked:** `{t['tool']}`\n- **Parameters:** `{t['args']}`")
                        elif t.get("type") == "result":
                            snippet = str(t['content'])[:350] + ("..." if len(str(t['content'])) > 350 else "")
                            st.markdown(f"**📥 Result from `{t['tool']}`:**\n```text\n{snippet}\n```")
                        st.markdown("---")

    # User Input Prompt Box
    if prompt := st.chat_input("Ask about Singapore travel, live weather (including specific dates), or currency conversion..."):
        # Display user query in UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Validate HuggingFace API Token
        if not hf_token:
            warning_msg = (
                "⚠️ **Hugging Face API Token missing!**\n\n"
                "Please configure `HUGGINGFACEHUB_API_TOKEN` in your `.env` file in the project root to execute queries.\n\n"
                "*(See `README.md` or `.env.example` for instructions)*"
            )
            st.session_state.messages.append({"role": "assistant", "content": warning_msg})
            with st.chat_message("assistant"):
                st.markdown(warning_msg)
            return

        # Execute Agent query with spinner
        with st.chat_message("assistant"):
            with st.spinner("Analyzing query, searching knowledge base, and calling live tools..."):
                # Execute async function safely inside Streamlit loop
                loop = asyncio.get_event_loop()
                response_text, tool_traces = loop.run_until_complete(
                    execute_agent_query(
                        user_input=prompt,
                        hf_token=hf_token,
                        chat_history=st.session_state.messages[:-1],
                        model_name=selected_model
                    )
                )
                st.markdown(response_text)
                if tool_traces:
                    with st.expander("🛠️ Tool Executions & Retrieved Citations", expanded=False):
                        for t in tool_traces:
                            if t.get("type") == "call":
                                st.markdown(f"**⚡ Tool Invoked:** `{t['tool']}`\n- **Parameters:** `{t['args']}`")
                            elif t.get("type") == "result":
                                snippet = str(t['content'])[:350] + ("..." if len(str(t['content'])) > 350 else "")
                                st.markdown(f"**📥 Result from `{t['tool']}`:**\n```text\n{snippet}\n```")
                            st.markdown("---")
                
        # Store assistant response in session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "tool_traces": tool_traces
        })


if __name__ == "__main__":
    main()
