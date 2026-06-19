"""RagGem - hardened RAG service.

Public surface:
  - :class:`raggem.core.engine.RagEngine` - the RAG pipeline.
  - :mod:`raggem.core.brains` - brain CRUD on disk.
  - :mod:`raggem.server` - FastAPI app, served with ``uvicorn raggem.server:app``.
"""

from .config import APP_NAME, APP_VERSION

__all__ = ["APP_NAME", "APP_VERSION"]
