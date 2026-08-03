"""MCP server exposing RAG search over the PyColmap API dataset to a
coding agent. Thin protocol layer only - all the actual search logic
lives in search.py so it stays usable outside of MCP too.
"""

import sys
from pathlib import Path

from mcp.server import MCPServer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "config"))
from config import load_llm_api_key

sys.path.insert(0, str(Path(__file__).resolve().parent))
import search as rag_search

mcp = MCPServer(name="pycolmap-rag")


@mcp.tool()
def search_pycolmap_api(query: str, top_k: int = 5) -> list[dict]:
    """Search the PyColmap API for classes, functions, methods, or
    properties matching a natural-language or code-shaped query. Returns
    each match's full API record (signatures, parameters, explanation)."""
    return rag_search.search(query, mcp_state["collection"], mcp_state["records_by_id"], mcp_state["api_key"], top_k=top_k)


def load_state():
    import pycolmap

    version = pycolmap.__version__
    return {
        "collection": rag_search.get_collection(version),
        "records_by_id": rag_search.load_records_by_id(version),
        "api_key": load_llm_api_key(),
    }


mcp_state = None


def main():
    global mcp_state
    mcp_state = load_state()
    mcp.run()


if __name__ == "__main__":
    main()
