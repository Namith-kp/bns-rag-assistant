from mcp.server.fastmcp import FastMCP
from rag_pipeline import ask, ConversationState

mcp = FastMCP("bns_2023_rag")

@mcp.tool()
def ask_bns(query: str, history: list[dict] = None) -> str:
    """
    Query the BNS 2023 legal database.
    Runs fully offline with deterministic follow-up resolution via ConversationState.

    Args:
        query: The question to ask about the BNS 2023.
        history: Optional list of previous conversation turns
                 e.g. [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    try:
        state = ConversationState.from_history_list(history or [])
        answer, sources = ask(query, state=state)

        result = answer + "\n\nSources:\n"
        for i, source in enumerate(sources, 1):
            result += f"[{i}] {source[:200]}...\n"

        return result
    except Exception as e:
        return f"Error querying BNS 2023 database: {e}"

if __name__ == "__main__":
    mcp.run()
