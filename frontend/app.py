"""Streamlit dashboard entry point — implemented in TICKET-14."""
import sys
from pathlib import Path

# Make the backend package (`src`) importable from the frontend.
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))
