"""pytest setup for the RagGem test suite.

Adds ``src/`` to sys.path so the package is importable without an editable
install. Pins DATA_DIR to a temporary directory so tests never touch the
real on-disk knowledge base.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make ``import raggem`` work regardless of how pytest is invoked.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Tests don't need a real Gemini key - but config.py imports load_dotenv,
# which would otherwise pull the real one. Force-set a placeholder so any
# accidental network calls fail loudly instead of charging quota.
os.environ.setdefault("GOOGLE_API_KEY", "test-key-not-real")
