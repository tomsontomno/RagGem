"""Optional MCP (Model Context Protocol) stdio server.

This makes the same operations the HTTP API exposes available to MCP
clients (Claude Desktop, Claude Code, any MCP-aware agent) over stdio
JSON-RPC. The dependency is optional: install ``mcp`` only if you want to
use ``raggem mcp``.

Tools exposed:
  - rag_query    : ask a question of a brain
  - rag_upload   : upload a file by path
  - rag_delete   : delete a file
  - rag_list     : list files in a brain
  - rag_brains   : list all brains
  - rag_rebuild  : rebuild a brain's vector store
  - rag_destroy  : destroy a brain (requires confirm flag)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


_INSTALL_HINT = (
    "MCP support is optional. Install it with:\n"
    "    pip install 'mcp>=1.0'\n"
    "Then re-run `raggem mcp`."
)


def run_stdio() -> None:
    """Run the MCP server on stdio.

    Raises:
        RuntimeError: if the ``mcp`` package is not installed.
    """
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc

    from .core.brains import (
        delete_file,
        destroy_brain as _destroy_brain,
        list_brains,
        list_files,
        save_uploaded_file,
    )
    from .core.engine import (
        get_engine,
        invalidate_brain,
        query_brain,
        remove_source,
    )

    server = FastMCP("raggem")

    @server.tool()
    def rag_brains() -> Dict[str, Any]:
        """List all known brains."""
        return {"brains": list_brains()}

    @server.tool()
    def rag_list(brain_id: str) -> Dict[str, Any]:
        """List files in a brain.

        Args:
            brain_id: brain identifier.
        """
        return {
            "brain_id": brain_id,
            "files": [
                {
                    "filename": f.filename,
                    "size_bytes": f.size_bytes,
                    "extension": f.extension,
                }
                for f in list_files(brain_id)
            ],
        }

    @server.tool()
    def rag_query(brain_id: str, question: str) -> Dict[str, Any]:
        """Ask a question of a brain.

        Args:
            brain_id: brain identifier.
            question: natural-language question.
        """
        return query_brain(brain_id, question)

    @server.tool()
    def rag_upload(brain_id: str, path: str) -> Dict[str, Any]:
        """Upload a local file into a brain and index it.

        Args:
            brain_id: brain identifier.
            path: absolute path to the source file on the machine running
                the MCP server.
        """
        src = Path(path).expanduser()
        if not src.is_file():
            raise FileNotFoundError(f"{path!r} is not a file")
        with src.open("rb") as fh:
            stored = save_uploaded_file(brain_id, src.name, fh)
        chunks = get_engine(brain_id).index_file(stored.name)
        invalidate_brain(brain_id)
        return {
            "brain_id": brain_id,
            "filename": stored.name,
            "chunks_indexed": chunks,
        }

    @server.tool()
    def rag_delete(brain_id: str, filename: str) -> Dict[str, Any]:
        """Delete a single file from a brain.

        Args:
            brain_id: brain identifier.
            filename: filename inside the brain (no path).
        """
        chunks_removed = remove_source(brain_id, filename)
        file_removed = delete_file(brain_id, filename)
        invalidate_brain(brain_id)
        return {
            "brain_id": brain_id,
            "filename": filename,
            "file_removed": file_removed,
            "chunks_removed": chunks_removed,
        }

    @server.tool()
    def rag_rebuild(brain_id: str) -> Dict[str, Any]:
        """Drop the vector store and reindex every file in a brain."""
        from .core.brains import reset_vector_store

        reset_vector_store(brain_id)
        invalidate_brain(brain_id)
        counts = get_engine(brain_id).index_all()
        return {
            "brain_id": brain_id,
            "reindexed": counts,
            "total_chunks": sum(counts.values()),
        }

    @server.tool()
    def rag_destroy(brain_id: str, confirm: bool = False) -> Dict[str, Any]:
        """Destroy a brain entirely. Requires ``confirm=true``."""
        if not confirm:
            return {
                "brain_id": brain_id,
                "destroyed": False,
                "error": "confirm=true required to destroy a brain",
            }
        _destroy_brain(brain_id)
        invalidate_brain(brain_id)
        return {"brain_id": brain_id, "destroyed": True}

    logger.info("Starting MCP stdio server (tools: 7)")
    server.run()
