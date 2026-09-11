"""
config.py – Centralised configuration for the EDA Streamlit app.

Resolution order for every setting:
  1. st.secrets
  2. os.environ (loaded from .env)
  3. Hard-coded default
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv


# Load .env from the SAME folder as this config.py file
BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)


def _get_secret(key: str, default: str | None = None) -> str | None:
    """
    Return a config value by checking st.secrets first,
    then environment variables, then the default.
    """

    # 1. Try Streamlit secrets
    try:
        import streamlit as st

        value = st.secrets.get(key)

        if value is not None:
            return str(value)

    except Exception:
        pass

    # 2. Environment variable
    value = os.getenv(key)

    if value is not None:
        return value

    # 3. Default
    return default


# Public configuration
XAI_API_KEY: str | None = _get_secret("XAI_API_KEY")

XAI_BASE_URL: str = "https://api.x.ai/v1"

XAI_TEXT_MODEL: str = "grok-4-latest"

XAI_IMAGE_MODEL: str = "grok-imagine-image"