"""
Model Context Protocol (MCP) External Tools Server for AI Travel Planning Assistant.
"""

from mcp_server import mcp, convert_currency, get_weather

if __name__ == "__main__":
    mcp.run(transport="stdio")
