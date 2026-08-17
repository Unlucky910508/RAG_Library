"""MCP server exposing RAG search over an already-built dataset to a
coding agent. Thin protocol layer only - all the actual search logic
lives in search.py so it stays usable outside of MCP too.

This directory stands alone: it reads nothing from the pipeline's config,
imports nothing from the rest of the tree, and does not need the library
it describes to be installed. Copy it somewhere with a Chroma store, point mcp_server_config.py at
that store, and run this - the store carries the answers as well as the
vectors, so nothing else travels with it.
"""

import sys
from pathlib import Path
from typing import Annotated

from mcp.server import MCPServer
from pydantic import Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
import search as rag_search
from mcp_server_config import API_KEY, LIBRARY_NAME, MAX_TOP_K, SERVER_NAME

mcp = MCPServer(name=SERVER_NAME)


# Stated as a description rather than a Field(le=...) bound: a bound makes
# the SDK reject an over-limit call outright, where clamping still answers
# it. The description carries the limit into the tool's JSON schema, so a
# calling model sees it without having to read the docstring.
TopK = Annotated[
    int,
    Field(description=f"How many results to return. Ask for at most {MAX_TOP_K}; larger values are reduced to {MAX_TOP_K}."),
]


# Passed to the decorator rather than left as a docstring so the limit can
# be interpolated from config - the decorator reads the description at
# registration time, so assigning __doc__ afterwards would not reach the
# tool the model sees.
SEARCH_DESCRIPTION = (
    f"Search the {LIBRARY_NAME} API and official example code for entries "
    "matching a natural-language or code-shaped query. Returns up to "
    f"{MAX_TOP_K} matches, each with the text for that kind of match: API "
    "signatures and explanation, or example source code.\n\n"
    "Keep top_k small - every hit carries a full record's worth of text, so "
    "asking for more than you need floods your context with loosely-related "
    f"matches. The maximum is {MAX_TOP_K}."
)


@mcp.tool(description=SEARCH_DESCRIPTION)
def search_rag(query: str, top_k: TopK = MAX_TOP_K) -> list[dict]:
    return rag_search.search(query, mcp_state["collection"], mcp_state["api_key"], top_k=top_k)


def load_state():
    """Opened once at startup rather than per query."""
    return {
        "collection": rag_search.get_collection(),
        "api_key": API_KEY,
    }


mcp_state = None


def main():
    global mcp_state
    mcp_state = load_state()
    mcp.run()


if __name__ == "__main__":
    main()
