"""RAG engine: retrieval + grounding gate + generation.

No LangChain. Uses google.genai for embeddings and generation, chromadb
for vector storage.

Architecture:
  - embedding and storage  -> :mod:`raggem.core.ingest`
  - path safety            -> :mod:`raggem.core.brains`
  - prompt construction    -> :mod:`raggem.prompts`
  - input validation       -> :mod:`raggem.security`

Per-brain ``RagEngine`` instances are cached; invalidate after any write.

Hoare contract for ``RagEngine.query``:
  Precondition  : ``brain_id`` validated; ``question`` sanitised;
                  ``GOOGLE_API_KEY`` set in environment.
  Postcondition : dict with keys ``answer``, ``sources``, ``metadata``.
                  ``answer`` is either a grounded reply or the fixed
                  out-of-scope refusal. No system-prompt content leaks.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import google.genai as genai
import google.genai.types as genai_types

from ..config import (
    EMBEDDING_MODEL,
    GROUNDING_MAX_DISTANCE,
    MAX_OUTPUT_TOKENS,
    MODEL_NAME,
    RETRIEVAL_K,
    TEMPERATURE,
    THINKING_BUDGET,
)
from ..prompts import (
    REFUSAL_OUT_OF_SCOPE,
    build_prompt,
    build_verify_prompt,
    is_refusal,
    parse_supported_ids,
    postprocess_answer,
)
from ..security import sanitize_question
from .brains import brain_paths, list_files
from .ingest import (
    Document,
    chunk_documents,
    delete_source_from_store,
    embed_and_store,
    load_file,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal hit record
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Hit:
    source: str
    page: int
    content: str
    distance: float
    url: str = ""
    title: str = ""


# ---------------------------------------------------------------------------
# Generation retry settings - for transient 503 / UNAVAILABLE responses.
# Embedding retries live in ingest.py; these cover only the generation step.
_GENERATE_MAX_RETRIES: int = 3
_GENERATE_BASE_DELAY_S: int = 5

# Max characters of a source's content returned as a citation snippet. Sources
# are meant to be short subpage links + a hint, not the full record dump.
_SOURCE_SNIPPET_CHARS: int = 160


# Engine error
# ---------------------------------------------------------------------------
class RagEngineError(RuntimeError):
    """Raised when the engine cannot fulfil a query for non-user reasons.

    For user-facing problems (out-of-scope question, empty brain) we
    return a grounded refusal instead of raising.
    """


# ---------------------------------------------------------------------------
# RAG engine
# ---------------------------------------------------------------------------
class RagEngine:
    """Per-brain RAG pipeline.

    Holds a google.genai Client and a lazy chromadb Collection handle.
    The collection is opened on first access and reused across queries.
    """

    def __init__(self, brain_id: str) -> None:
        if not os.environ.get("GOOGLE_API_KEY", "").strip():
            raise RagEngineError("GOOGLE_API_KEY is not set in the environment.")
        self.brain_id = brain_id
        self._paths = brain_paths(brain_id)
        self._client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        self._collection: Optional[chromadb.Collection] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Vector store lifecycle
    # ------------------------------------------------------------------
    def _open_collection(self) -> chromadb.Collection:
        """Open (or create) the chromadb collection for this brain.

        Postcondition: returns a Collection backed by a persistent directory;
        the directory is created if absent.
        """
        self._paths.chroma_dir.mkdir(parents=True, exist_ok=True)
        db = chromadb.PersistentClient(path=str(self._paths.chroma_dir))
        return db.get_or_create_collection("knowledge")

    @property
    def vector_store(self) -> chromadb.Collection:
        if self._collection is None:
            with self._lock:
                if self._collection is None:
                    self._collection = self._open_collection()
        return self._collection

    def reset_store(self) -> None:
        """Drop the cached collection handle so the next access reopens it."""
        with self._lock:
            self._collection = None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def index_file(self, filename: str) -> int:
        """Embed one source file into the brain's vector store.

        Removes any pre-existing chunks for the same source first so
        re-uploads don't accumulate duplicates.

        Args:
            filename: filename inside the brain's knowledge dir.

        Returns:
            Number of chunks added.
        """
        path = self._paths.knowledge_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"{filename!r} not found in brain {self.brain_id!r}")
        docs = load_file(path)
        # JSON files are pre-split into one Document per record by load_file;
        # running chunk_documents on top would break records across chunk
        # boundaries and corrupt the structured metadata.
        if path.suffix.lower() == ".json":
            chunks = docs
        else:
            chunks = chunk_documents(docs)
        delete_source_from_store(self.vector_store, filename)
        return len(embed_and_store(self.vector_store, chunks, self._client))

    def index_all(self) -> Dict[str, int]:
        """Reindex every supported document file in the brain.

        Returns:
            Mapping of filename -> number of chunks added.
        """
        counts: Dict[str, int] = {}
        for info in list_files(self.brain_id):
            counts[info.filename] = self.index_file(info.filename)
        return counts

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def _embed_query(self, text: str) -> List[float]:
        """Embed a single query string.

        Precondition:  ``text`` is non-empty and sanitised.
        Postcondition: returns a float list (the query embedding vector).
        """
        resp = self._client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
        )
        return list(resp.embeddings[0].values)

    def _retrieve(self, question: str) -> List[_Hit]:
        """Run a similarity search and return sorted hits.

        Returns the top-k hits as ``_Hit`` records sorted by distance
        ascending (closest match first). An empty list means the brain
        is empty or the search failed.
        """
        try:
            q_vec = self._embed_query(question)
            results = self.vector_store.query(
                query_embeddings=[q_vec],
                n_results=RETRIEVAL_K,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            logger.exception("Vector search failed for brain %r", self.brain_id)
            return []

        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]

        hits = [
            _Hit(
                source=str((meta or {}).get("source", "unknown")),
                page=int((meta or {}).get("page", 0)),
                content=str(doc or ""),
                distance=float(dist),
                url=str((meta or {}).get("url", "")),
                title=str((meta or {}).get("title", "")),
            )
            for doc, meta, dist in zip(docs, metas, dists)
        ]
        hits.sort(key=lambda h: h.distance)
        return hits

    def _is_grounded(self, hits: List[_Hit]) -> bool:
        """True iff retrieval produced at least one hit within the distance threshold."""
        return bool(hits) and hits[0].distance <= GROUNDING_MAX_DISTANCE

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _generate(self, prompt: str) -> str:
        """Call the generation model and return the text response.

        Retries up to _GENERATE_MAX_RETRIES times on transient 503 /
        UNAVAILABLE responses with exponential backoff. Hard errors (4xx,
        auth failures) are never retried.

        Precondition:  ``prompt`` is a non-empty string; ``self._client``
                       is initialised with a valid API key.
        Postcondition: returns a str (empty string if the model produced
                       no text content); raises the original exception after
                       all retries are exhausted, with context preserved.
        """
        config: Dict[str, Any] = {
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        }
        if THINKING_BUDGET != 0:
            config["thinking_config"] = genai_types.ThinkingConfig(
                thinking_budget=THINKING_BUDGET
            )
        gen_config = genai_types.GenerateContentConfig(**config)

        for attempt in range(_GENERATE_MAX_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=gen_config,
                )
                return response.text or ""
            except Exception as exc:
                msg = str(exc)
                is_transient = any(
                    needle in msg.lower()
                    for needle in ("503", "unavailable", "overloaded", "capacity")
                )
                if not is_transient or attempt == _GENERATE_MAX_RETRIES - 1:
                    raise
                delay = _GENERATE_BASE_DELAY_S * (2 ** attempt)
                logger.warning(
                    "Generation server unavailable (attempt %d/%d); retrying in %ds. %s",
                    attempt + 1,
                    _GENERATE_MAX_RETRIES,
                    delay,
                    msg.splitlines()[0][:120],
                )
                time.sleep(delay)
        return ""  # unreachable: the loop always raises or returns inside

    # ------------------------------------------------------------------
    # Source verification (re-challenge)
    # ------------------------------------------------------------------
    def _verify(self, answer: str, hits: List[_Hit]) -> List[int]:
        """Return the 1-based hit numbers that actually support ``answer``.

        A second, cheap LLM pass re-reads the answer against the retrieved
        sources and names only the ones that back its claims, so the caller can
        cite what genuinely supports the answer instead of every retrieved
        chunk.

        Precondition:  ``answer`` is the post-processed, non-refusal answer;
                       ``hits`` is the retrieved list.
        Postcondition: returns 1-based indices into ``hits`` (retrieval order).
                       Returns [] when ``hits`` is empty or the verification
                       call fails, so unverified sources are never attached.
        """
        if not hits:
            return []
        candidates = [(h.title or h.source, h.content) for h in hits]
        try:
            reply = self._generate(build_verify_prompt(answer, candidates))
        except Exception:
            logger.exception("Source verification failed for brain %r", self.brain_id)
            return []
        return parse_supported_ids(reply, len(hits))

    # ------------------------------------------------------------------
    # Public query
    # ------------------------------------------------------------------
    def query(self, question: str) -> Dict[str, Any]:
        """Answer a question from this brain's knowledge.

        Args:
            question: raw question from the user; will be sanitised.

        Returns:
            ``{"answer": str, "sources": [...], "metadata": {...}}``.
            Out-of-scope answers return the canonical refusal with empty
            sources; no system-prompt content leaks in either case.
        """
        clean_q = sanitize_question(question)
        hits = self._retrieve(clean_q)

        if not self._is_grounded(hits):
            return {
                "answer": REFUSAL_OUT_OF_SCOPE,
                "sources": [],
                "metadata": {
                    "brain_id": self.brain_id,
                    "model_used": MODEL_NAME,
                    "retrieved_chunks": len(hits),
                    "grounded": False,
                    "best_distance": hits[0].distance if hits else None,
                },
            }

        prompt = build_prompt(
            chunks=[(h.source, h.content) for h in hits],
            question=clean_q,
        )

        try:
            raw_answer = self._generate(prompt)
        except Exception as exc:
            msg = str(exc)
            short = (
                "rate-limited by the model provider; try again later"
                if any(
                    needle in msg.lower()
                    for needle in ("429", "rate", "quota", "resource_exhausted")
                )
                else f"LLM call failed: {msg.splitlines()[0][:160]}"
            )
            raise RagEngineError(short) from exc

        answer = postprocess_answer(raw_answer)

        # Re-challenge: attach only sources that actually support the answer.
        # A refusal cites nothing - this stops "not found" replies from still
        # listing the top retrieved chunks.
        supported = [] if is_refusal(answer) else self._verify(answer, hits)

        sources = [
            {
                "source_id": n,
                "title": hits[n - 1].title or hits[n - 1].source,
                "url": hits[n - 1].url,
                "source": hits[n - 1].source,
                "page": hits[n - 1].page,
                "content": hits[n - 1].content[:_SOURCE_SNIPPET_CHARS],
                "distance": hits[n - 1].distance,
            }
            for n in supported
        ]

        return {
            "answer": answer,
            "sources": sources,
            "metadata": {
                "brain_id": self.brain_id,
                "model_used": MODEL_NAME,
                "retrieved_chunks": len(hits),
                "cited_sources": len(sources),
                "grounded": True,
                "best_distance": hits[0].distance,
            },
        }


# ---------------------------------------------------------------------------
# Per-brain engine cache
# ---------------------------------------------------------------------------
_ENGINES: Dict[str, RagEngine] = {}
_ENGINES_LOCK = threading.Lock()


def get_engine(brain_id: str) -> RagEngine:
    """Return a cached ``RagEngine`` for ``brain_id``, creating one if needed."""
    with _ENGINES_LOCK:
        engine = _ENGINES.get(brain_id)
        if engine is None:
            engine = RagEngine(brain_id)
            _ENGINES[brain_id] = engine
        return engine


def invalidate_brain(brain_id: str) -> None:
    """Drop the cached engine after any write operation on the brain."""
    with _ENGINES_LOCK:
        engine = _ENGINES.pop(brain_id, None)
    if engine is not None:
        engine.reset_store()
        logger.info("Invalidated engine cache for brain %r", brain_id)


def index_one(brain_id: str, filename: str) -> int:
    """Reindex a single file. Convenience wrapper used by the CLI and API."""
    return get_engine(brain_id).index_file(filename)


def remove_source(brain_id: str, filename: str) -> int:
    """Remove all chunks for a source file. Convenience wrapper."""
    return delete_source_from_store(get_engine(brain_id).vector_store, filename)


_QUERY_FALLBACK: Tuple[str, str] = (
    REFUSAL_OUT_OF_SCOPE,
    "no knowledge configured",
)


def query_brain(brain_id: str, question: str) -> Dict[str, Any]:
    """Convenience: get the engine and run a query in one call."""
    return get_engine(brain_id).query(question)
