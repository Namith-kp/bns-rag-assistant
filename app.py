"""
app.py — CLI loop for the BNS 2023 RAG assistant.

On startup, verifies that the required LM Studio models are loaded before
attempting any embeddings or chat completions.
"""

import sys
import requests

from rag_pipeline import ask, ConversationState

# ── Config ────────────────────────────────────────────────────────────────────
LMSTUDIO_BASE  = "http://localhost:1234"
REQUIRED_CHAT  = "qwen2.5-0.5b-instruct"
REQUIRED_EMBED = "text-embedding-nomic-embed-text-v1.5"


# ── Startup check ─────────────────────────────────────────────────────────────
def check_lmstudio_models() -> None:
    """Abort with a clear message if required models are not loaded in LM Studio."""
    print("[app] Checking LM Studio models...")
    try:
        resp = requests.get(f"{LMSTUDIO_BASE}/v1/models", timeout=5)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        sys.exit(
            "[app] ERROR: Cannot connect to LM Studio at http://localhost:1234. "
            "Please start LM Studio and load the required models."
        )
    except Exception as e:
        sys.exit(f"[app] ERROR: Unexpected error contacting LM Studio: {e}")

    loaded_ids = {m["id"] for m in resp.json().get("data", [])}

    missing = []
    if REQUIRED_CHAT  not in loaded_ids:
        missing.append(f"  - Chat model    : {REQUIRED_CHAT}")
    if REQUIRED_EMBED not in loaded_ids:
        missing.append(f"  - Embed model   : {REQUIRED_EMBED}")

    if missing:
        sys.exit(
            "[app] ERROR: The following models are not loaded in LM Studio:\n"
            + "\n".join(missing)
            + "\nPlease load them and restart this app."
        )

    print(f"[app] [OK] LM Studio OK - chat='{REQUIRED_CHAT}', embed='{REQUIRED_EMBED}'")


# ── CLI loop ──────────────────────────────────────────────────────────────────
def main() -> None:
    check_lmstudio_models()

    print("\n=== BNS 2023 RAG Assistant ===")
    print("Type your question and press Enter. Type 'exit' to quit.\n")

    # ConversationState tracks history + current_topic for follow-up resolution
    state = ConversationState()

    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[app] Exiting.")
            break

        if not query:
            continue
        if query.lower() == "exit":
            print("[app] Goodbye!")
            break

        print("\n[app] Retrieving & generating answer...")
        try:
            answer, sources = ask(query, state=state)
        except Exception as e:
            print(f"[app] ERROR during RAG pipeline: {e}\n")
            continue

        # Update state with this turn's result
        state.add_turn("user", query)
        state.add_turn("assistant", answer)

        print(f"\n--- Answer ---\n{answer}\n")
        print("--- Source excerpts used ---")
        for i, src in enumerate(sources, 1):
            print(f"\n[{i}] {src[:300].strip()}{'...' if len(src) > 300 else ''}")
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
