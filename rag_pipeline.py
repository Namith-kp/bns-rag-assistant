"""
rag_pipeline.py — Query the ChromaDB collection and answer via LM Studio chat.

Pipeline:
  1. ConversationState   — tracks history and current topic for follow-up resolution
  2. Query expansion     — augments short/vague queries with topic synonyms
  3. Wide vector fetch   — TOP_K_FETCH=20 candidates from ChromaDB
  4. Re-rank             — composite score: keyword overlap × legal-content quality
                           TOC chunks are penalised; definition chunks rewarded
  5. Similarity threshold — declines to answer if top score < RETRIEVAL_THRESHOLD
  6. LLM generation      — system+user split, fill-in-the-blank prompt style
"""

import re
import sys
import chromadb
from openai import OpenAI

# Ensure the terminal/log can handle Unicode characters on Windows
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH        = "./chroma_db"
COLLECTION_NAME    = "bns_2023"
EMBED_MODEL        = "text-embedding-nomic-embed-text-v1.5"
CHAT_MODEL         = "qwen2.5-0.5b-instruct"
BASE_URL           = "http://localhost:1234/v1"
API_KEY            = "lm-studio"

TOP_K_FETCH        = 20    # wide net from vector DB
TOP_K_USE          = 4     # passages sent to LLM (small model saturates with more)
MAX_CHUNK_CHARS    = 480   # per-chunk display cap
MAX_CONTEXT_CHARS  = 1900  # total context cap
TEMPERATURE        = 0.05  # near-deterministic for factual Q&A
MAX_TOKENS         = 450

# Minimum composite score for the top result to be considered "on-topic".
# Queries whose best chunk falls below this are declined rather than answered
# from weak matches (avoids hallucination on out-of-scope topics).
RETRIEVAL_THRESHOLD = 2.0

# ── Legal signal phrases ──────────────────────────────────────────────────────
_LEGAL_SIGNALS = [
    "shall be punished", "punishable with", "liable to fine", "with fine",
    "rigorous imprisonment", "simple imprisonment", "imprisonment for life",
    "imprisonment of either description", "shall extend", "not less than",
    "not exceed", "death",
    "whoever", "any person who", "no person shall", "shall not",
    "notwithstanding", "provided that", "explanation",
    "convicted", "sentence", "offence",
    "is said to", "means", "includes", "does not include",
    "dishonestly", "fraudulently", "intentionally", "voluntarily",
    "moveable property", "consent", "without consent",
]

# A pure TOC line: "NNN. Some Title." — no sentence structure after the number
_TOC_LINE = re.compile(r"^\d{1,3}\.\s+[A-Z][^\.\n]{5,80}[.\-]?\s*$")

# ── Query expansion map ───────────────────────────────────────────────────────
_EXPANSION = {
    "theft":             "dishonestly takes moveable property without consent section 303",
    "murder":            "culpable homicide death punishment section 103",
    "rape":              "sexual assault penetration consent punishment section 64",
    "robbery":           "theft extortion force section 309",
    "dacoity":           "robbery five persons section 310",
    "assault":           "criminal force apprehension hurt section 131",
    "cheating":          "fraud deception property section 318",
    "extortion":         "induces delivery property threat section 308",
    "kidnapping":        "abduction minor lawful guardian section 137",
    "abduction":         "kidnapping compels section 138",
    "defamation":        "imputation reputation publication section 356",
    "sedition":          "sovereignty integrity India section 147 152",
    "forgery":           "false document electronic record section 335",
    "dowry death":       "harassment cruelty husband relatives section 80",
    "hurt":              "bodily pain disease infirmity section 114",
    "grievous hurt":     "fracture dislocation permanent section 116",
    "culpable homicide": "not amounting murder death bodily injury section 105",
    "attempt":           "offence endeavours fails section 62",
    "conspiracy":        "agreement criminal act section 61",
    "abetment":          "instigates aids facilitates section 45",
}

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a legal assistant specialising in the Bharatiya Nyaya Sanhita (BNS) 2023.
Rules:
1. Answer ONLY from the numbered passages provided below the question.
2. Always state the section number and the exact penalty/punishment wording.
3. If the passages contain a relevant answer, provide it directly and completely — do NOT append "Not found in the provided excerpts" after a real answer.
4. ONLY say "Not found in the provided excerpts." when NONE of the passages contain any information relevant to the question.
5. Do not invent, infer, or use outside knowledge.\
"""

# Out-of-scope decline message (returned when top retrieval score is too low)
_NOT_COVERED = (
    "This isn't covered in the indexed BNS 2023 text. "
    "The Bharatiya Nyaya Sanhita deals with criminal offences; "
    "your question appears to fall outside that scope."
)

# ── Shared clients ────────────────────────────────────────────────────────────
_openai_client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
_chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

def _get_collection():
    return _chroma_client.get_collection(name=COLLECTION_NAME)


# ── ConversationState ─────────────────────────────────────────────────────────

# Reference words that signal the query is a pronoun-style follow-up
_REFERENCE_WORDS = frozenset({
    "it", "its", "this", "that", "these", "those", "same", "above",
    "they", "them", "their", "he", "she", "him", "her",
})

# Stop words excluded when counting "content words"
_CONTENT_STOP = frozenset({
    "what", "is", "the", "of", "for", "a", "an", "in", "and", "or",
    "under", "bns", "2023", "how", "many", "does", "do", "to", "by",
    "which", "who", "when", "where", "section", "act", "provision",
    "define", "definition", "meaning", "explain", "are", "can",
    "tell", "me", "about", "give", "detail",
})


def is_low_content_query(query: str) -> bool:
    """
    Returns True if the query is a pronoun-style follow-up that needs
    conversation history to be meaningful.  Two triggers:
      1. The query contains a reference word (it, this, that, those ...)
      2. The query has <= 1 content word longer than 3 chars
    """
    words = re.findall(r"[a-z]+", query.lower())
    if any(w in _REFERENCE_WORDS for w in words):
        return True
    content = [w for w in words if len(w) > 3 and w not in _CONTENT_STOP]
    return len(content) <= 1


class ConversationState:
    """
    Tracks conversation history and the current legal topic.

    current_topic is set to the last query that:
      - is NOT a low-content follow-up, AND
      - returned a confident retrieval (top score >= RETRIEVAL_THRESHOLD)

    For low-content follow-ups, the topic is prepended to the raw query
    before embedding so retrieval stays anchored to the right section.
    """

    def __init__(self, history: list[dict] = None):
        # history is a list of {"role": "user"/"assistant", "content": "..."}
        self.history: list[dict] = list(history) if history else []
        # Infer current_topic from history (last non-follow-up user turn)
        self.current_topic: str | None = None
        for turn in self.history:
            if turn.get("role") == "user":
                q = turn["content"]
                if not is_low_content_query(q):
                    self.current_topic = q

    def add_turn(self, role: str, content: str) -> None:
        """Append a turn to history."""
        self.history.append({"role": role, "content": content})

    def update_topic(self, query: str) -> None:
        """Record query as the new current topic (call when retrieval was confident)."""
        self.current_topic = query

    def get_retrieval_query(self, query: str) -> str:
        """
        Return the query string to use for embedding + retrieval.

        - Self-contained query   → return as-is
        - Follow-up with topic   → prepend current_topic
        - Follow-up without topic → return as-is (no history to anchor on)
        """
        if not is_low_content_query(query):
            return query
        if self.current_topic:
            return f"{self.current_topic} {query}"
        return query

    @classmethod
    def from_history_list(cls, history: list[dict]) -> "ConversationState":
        """Construct a ConversationState from a raw history list."""
        return cls(history=history)


# ── Backward-compat shim ──────────────────────────────────────────────────────
def build_retrieval_query(current_query: str, history: list,
                          max_turns: int = 1) -> str:
    """Kept for backward compatibility. Prefer ConversationState.get_retrieval_query()."""
    state = ConversationState.from_history_list(history)
    return state.get_retrieval_query(current_query)


# ── Query expansion ───────────────────────────────────────────────────────────
def _expand_query(query: str) -> str:
    """
    Append domain-specific expansion terms for known legal concepts.
    Helps the embedding model find definition chunks that use different
    vocabulary than the query.
    """
    q_lower = query.lower()
    for keyword, expansion in _EXPANSION.items():
        if keyword in q_lower:
            return f"{query} {expansion}"
    return query


# ── Chunk scoring & filtering ─────────────────────────────────────────────────

def _toc_ratio(chunk: str) -> float:
    """Fraction of non-empty lines that look like pure TOC entries."""
    lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
    if not lines:
        return 1.0
    toc_count = sum(1 for ln in lines if _TOC_LINE.match(ln))
    return toc_count / len(lines)


def _legal_signal_score(chunk: str) -> float:
    """Count of legal-content signal phrases present (case-insensitive)."""
    lower = chunk.lower()
    return sum(1.0 for sig in _LEGAL_SIGNALS if sig in lower)


def _keyword_score(chunk: str, query: str) -> float:
    """
    Word-level overlap between chunk and query (stop-word filtered).
    Longer words are weighted slightly more.
    """
    stop = {
        "what", "is", "the", "of", "for", "a", "an", "in", "and", "or",
        "under", "bns", "2023", "how", "many", "does", "do", "to", "by",
        "which", "who", "when", "where", "section", "act", "provision",
        "define", "definition", "meaning", "explain",
    }
    words = [w for w in re.findall(r"[a-z]+", query.lower())
             if w not in stop and len(w) > 2]
    if not words:
        return 0.0
    lower = chunk.lower()
    return sum(lower.count(w) * (1.0 + len(w) * 0.05) for w in words)


def _composite_score(chunk: str, query: str) -> float:
    """
    score = keyword_overlap x (1 + legal_signals x 0.35)
    TOC-heavy chunks get x0.05 penalty.
    """
    toc = _toc_ratio(chunk)
    kw  = _keyword_score(chunk, query)

    if toc > 0.55:
        return kw * 0.05

    legal = _legal_signal_score(chunk)
    return kw * (1.0 + legal * 0.35)


def _rerank(candidates: list[str], query: str) -> list[str]:
    """Score, sort descending, return TOP_K_USE best chunks."""
    scored = sorted(candidates,
                    key=lambda c: _composite_score(c, query),
                    reverse=True)
    return scored[:TOP_K_USE]


# ── Context builder ───────────────────────────────────────────────────────────

def _build_user_message(query: str, sources: list[str]) -> str:
    """
    Numbered passages (best first) + question.
    Each chunk capped at MAX_CHUNK_CHARS; total capped at MAX_CONTEXT_CHARS.
    """
    passages_context = "\n\n".join(
        f"Passage {i+1}:\n{text[:MAX_CHUNK_CHARS]}"
        for i, text in enumerate(sources)
    )

    safe_query = query
    if len(query.split()) < 3 and "?" not in query:
        safe_query = f"What does the law say about {query}, including penalties?"

    return f"Passages:\n{passages_context}\n\nQuestion: {safe_query}\nAnswer:"


# ── Post-processing fallback ──────────────────────────────────────────────────

_WEAK_ANSWER_RE = re.compile(
    r"^(\d+\.\s.{0,80}|theft|murder|rape|robbery|assault|cheating|\w{1,25})$",
    re.IGNORECASE,
)


def _is_weak_answer(answer: str) -> bool:
    """True if the answer is a single token / bare section header."""
    stripped = answer.strip()
    return len(stripped) < 50 or bool(_WEAK_ANSWER_RE.fullmatch(stripped))


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _fallback_extract(query: str, sources: list[str]) -> str:
    """
    Extract the most query-relevant sentence(s) directly from the passages.
    Used when the LLM returns a trivially short or repetitive answer.
    """
    stop = {
        "what", "is", "the", "of", "for", "a", "an", "in", "and", "or",
        "under", "bns", "2023", "how", "does", "do", "to", "by", "which",
        "who", "when", "where", "define", "meaning",
    }
    words = [w for w in re.findall(r"[a-z]+", query.lower())
             if w not in stop and len(w) > 2]

    best_score = -1
    best_sentences: list[str] = []

    for chunk in sources:
        sentences = _SENT_SPLIT.split(chunk.replace("\n", " "))
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 40:
                continue
            lower = sent.lower()
            score = sum(lower.count(w) for w in words)
            if score > best_score:
                best_score = score
                best_sentences = [sent]
            elif score == best_score and score > 0:
                best_sentences.append(sent)

    if best_sentences:
        return "[Extracted from source] " + " ".join(best_sentences[:3])
    return "[Could not extract answer] Please read the source passages below."


# ── Public API ────────────────────────────────────────────────────────────────

def _embed(text: str) -> list[float]:
    response = _openai_client.embeddings.create(model=EMBED_MODEL, input=[text])
    return response.data[0].embedding


def ask(query: str,
        history: list = None,
        state: "ConversationState" = None) -> tuple[str, list[str]]:
    """
    Full RAG pipeline:
      ConversationState -> retrieval query -> expand -> embed -> fetch -> rerank
      -> similarity threshold -> LLM answer

    Parameters
    ----------
    query   : The user's raw question.
    history : Deprecated. Pass a ConversationState via `state` instead.
              If provided and `state` is None, a ConversationState is built
              from this list for backward compatibility.
    state   : ConversationState tracking history + current topic.

    Returns
    -------
    (answer_text, list_of_source_excerpts)
    """
    original_query = query

    # Build / reuse ConversationState
    if state is None:
        state = ConversationState.from_history_list(history or [])

    # Determine retrieval query from state (no LLM call)
    retrieval_query = state.get_retrieval_query(query)

    # ── Debug logging ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[ASK] original_query   : {original_query!r}")
    print(f"[ASK] retrieval_query  : {retrieval_query!r}")
    print(f"[ASK] current_topic    : {state.current_topic!r}")

    # 1. Expand retrieval query for embedding
    expanded = _expand_query(retrieval_query)
    query_embedding = _embed(expanded)

    # 2. Wide vector search
    results = _get_collection().query(
        query_embeddings=[query_embedding],
        n_results=TOP_K_FETCH,
        include=["documents"],
    )
    candidates: list[str] = results["documents"][0]

    # 3. Re-rank using retrieval_query keywords
    sources = _rerank(candidates, retrieval_query)

    # 4. Similarity threshold guard
    top_score = _composite_score(sources[0], retrieval_query) if sources else 0.0
    print(f"[ASK] top_score        : {top_score:.3f}  (threshold={RETRIEVAL_THRESHOLD})")
    if sources:
        print(f"[ASK] top_chunk_preview: {sources[0][:120].strip()!r}")
    print(f"{'='*60}\n")

    if top_score < RETRIEVAL_THRESHOLD:
        print(f"[ASK] Score below threshold — declining to answer.")
        # Don't update topic; don't add to state here (caller does that)
        return _NOT_COVERED, sources

    # 5. Update conversation topic (only when retrieval was confident)
    if not is_low_content_query(original_query):
        state.update_topic(original_query)

    # 6. Build prompt and call LLM (use original_query in the prompt)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": _build_user_message(original_query, sources)},
    ]
    response = _openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    answer = response.choices[0].message.content.strip()

    # 7. Fallback if model echoed a heading or gave a trivial reply
    if _is_weak_answer(answer):
        answer = _fallback_extract(original_query, sources)

    return answer, sources
