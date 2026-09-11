"""
grok_client.py – Thin wrapper around the xAI Grok API (OpenAI-compatible).

Features:
  - Automatic retry with exponential backoff for transient errors
  - Token budget guard to prevent context-window overflows
  - Every call is wrapped in try/except so a bad API response never crashes
    the Streamlit app
"""

from __future__ import annotations

import base64
import json
import logging
import time
import traceback
from typing import Any

from openai import OpenAI

from utils.prompts import (
    ANALYSIS_PLAN_SYSTEM,
    DASHBOARD_IMAGE_PROMPT_SYSTEM,
    EDA_CODE_SYSTEM,
    build_analysis_plan_user_message,
    build_dashboard_image_user_message,
    build_eda_code_user_message,
)

logger = logging.getLogger(__name__)

# Transient HTTP status codes that are safe to retry
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Maximum estimated tokens for the data profile (rough 4-chars-per-token heuristic)
_MAX_PROFILE_CHARS = 16_000  # ~4,000 tokens


class GrokClient:
    """
    Stateless client for xAI Grok text & image models.

    Intended to be created once per session via ``st.cache_resource``.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        text_model: str,
        image_model: str,
        max_retries: int = 3,
    ):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._text_model = text_model
        self._image_model = image_model
        self._max_retries = max_retries
        logger.info(
            "GrokClient initialised (text=%s, image=%s, retries=%d)",
            text_model, image_model, max_retries,
        )

    # ── Retry wrapper ────────────────────────────────────────────────────

    def _call_with_retry(self, fn, *args, **kwargs):
        """
        Call *fn* with automatic retry on transient API errors.
        Uses exponential backoff: 1s → 2s → 4s.
        """
        last_exc = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)

                # Check if the error response has a status code attribute
                if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
                    status = exc.response.status_code

                if status in _RETRYABLE_STATUS_CODES and attempt < self._max_retries:
                    wait = 2 ** (attempt - 1)  # 1s, 2s, 4s
                    logger.warning(
                        "Retryable error (status=%s, attempt %d/%d), "
                        "waiting %ds: %s",
                        status, attempt, self._max_retries, wait, exc,
                    )
                    time.sleep(wait)
                else:
                    raise
        raise last_exc  # type: ignore[misc]

    # ── Token budget guard ───────────────────────────────────────────────

    @staticmethod
    def _trim_profile(profile: dict[str, Any]) -> dict[str, Any]:
        """
        If the JSON-serialised profile is too large, trim sample_rows and
        categorical_top_values to fit within the token budget.
        Returns a (possibly trimmed) copy.
        """
        import copy
        p = copy.deepcopy(profile)
        serialised = json.dumps(p, default=str)

        if len(serialised) <= _MAX_PROFILE_CHARS:
            return p

        # Trim 1: reduce sample_rows to 3
        if "sample_rows" in p:
            p["sample_rows"] = p["sample_rows"][:3]

        # Trim 2: reduce categorical_top_values to top 5 per column
        if "categorical_top_values" in p:
            for col in p["categorical_top_values"]:
                items = list(p["categorical_top_values"][col].items())
                p["categorical_top_values"][col] = dict(items[:5])

        # Trim 3: remove top_correlations if still too large
        serialised = json.dumps(p, default=str)
        if len(serialised) > _MAX_PROFILE_CHARS:
            p.pop("top_correlations", None)
            p.pop("outlier_pct", None)

        # Trim 4: remove sample_rows entirely as last resort
        serialised = json.dumps(p, default=str)
        if len(serialised) > _MAX_PROFILE_CHARS:
            p["sample_rows"] = []

        logger.info(
            "Profile trimmed: %d → %d chars",
            len(json.dumps(profile, default=str)),
            len(json.dumps(p, default=str)),
        )
        return p

    # ── 1. Analysis plan ─────────────────────────────────────────────────

    def get_analysis_plan(
        self,
        profile: dict[str, Any],
        user_request: str,
    ) -> dict[str, Any]:
        """
        Ask Grok to decide *what* to analyse.

        Returns a dict with keys ``kpis``, ``charts``, ``insights``.

        Raises
        ------
        json.JSONDecodeError
            If Grok returns non-JSON content.
        Exception
            On any API / network error.
        """
        trimmed = self._trim_profile(profile)
        user_msg = build_analysis_plan_user_message(trimmed, user_request)

        def _call():
            return self._client.chat.completions.create(
                model=self._text_model,
                messages=[
                    {"role": "system", "content": ANALYSIS_PLAN_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )

        response = self._call_with_retry(_call)

        raw = response.choices[0].message.content or ""
        logger.debug("Raw analysis-plan response:\n%s", raw)

        plan = json.loads(raw)
        return plan

    # ── 2. EDA code generation ───────────────────────────────────────────

    def generate_eda_code(
        self,
        plan: dict[str, Any],
        profile: dict[str, Any],
        filename: str,
    ) -> str:
        """
        Ask Grok to write a self-contained EDA Python script.

        Returns the raw Python source code as a string.
        """
        user_msg = build_eda_code_user_message(plan, profile, filename)

        def _call():
            return self._client.chat.completions.create(
                model=self._text_model,
                messages=[
                    {"role": "system", "content": EDA_CODE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
            )

        response = self._call_with_retry(_call)

        code = response.choices[0].message.content or ""

        # Strip accidental markdown fences if Grok adds them anyway
        code = _strip_code_fences(code)

        logger.debug("Generated EDA code (%d chars)", len(code))
        return code

    # ── 3. Dashboard image ───────────────────────────────────────────────

    def generate_dashboard_image(
        self,
        kpi_results: list[dict[str, Any]],
        insights: list[str],
    ) -> bytes:
        """
        Two-step process:
          1. Text model turns KPIs + insights into an image-gen prompt.
          2. Image model generates the dashboard mockup.

        Returns raw PNG bytes.
        """
        # Step 1 – craft the image prompt via the text model
        user_msg = build_dashboard_image_user_message(kpi_results, insights)

        def _prompt_call():
            return self._client.chat.completions.create(
                model=self._text_model,
                messages=[
                    {"role": "system", "content": DASHBOARD_IMAGE_PROMPT_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.5,
            )

        prompt_response = self._call_with_retry(_prompt_call)
        image_prompt = (prompt_response.choices[0].message.content or "").strip()
        logger.info("Image-gen prompt: %s", image_prompt[:200])

        # Step 2 – generate the image
        def _image_call():
            return self._client.images.generate(
                model=self._image_model,
                prompt=image_prompt,
                n=1,
                response_format="b64_json",
            )

        image_response = self._call_with_retry(_image_call)

        b64_data = image_response.data[0].b64_json
        image_bytes = base64.b64decode(b64_data)
        logger.info("Dashboard image generated (%d bytes)", len(image_bytes))
        return image_bytes


# ── Private helpers ──────────────────────────────────────────────────────────

def _strip_code_fences(text: str) -> str:
    """Remove markdown ``` fences that Grok may add despite instructions."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)
