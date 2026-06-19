"""RagGem - hardened RAG service.

Public surface:
  - :class:`raggem.core.engine.RagEngine` - the RAG pipeline.
  - :mod:`raggem.core.brains` - brain CRUD on disk.
  - :mod:`raggem.server` - FastAPI app for HTTP integrations.
  - :mod:`raggem.cli` - argparse CLI exposed as ``raggem``.
"""

from .config import APP_NAME, APP_VERSION

__all__ = ["APP_NAME", "APP_VERSION"]
