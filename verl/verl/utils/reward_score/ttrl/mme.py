"""MME answer extraction and reward helpers for TTRV.

This module intentionally delegates to the repository-level shared MME
scorer so static baselines and live TTRV rewards use the same parsing rules.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


def _ensure_repo_root_on_path() -> None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "eval" / "mme_scoring.py").exists():
            parent_str = str(parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)
            return


_ensure_repo_root_on_path()

from src.eval.mme_scoring import (  # noqa: E402
    clean_model_response,
    contains_special_tokens,
    extract_mme_answer as _extract_mme_answer,
    normalise_mme_answer,
    normalise_yes_no,
    score_mme_answer,
    score_mme_output,
)


def _extra_value(extra_info: dict[str, Any] | None, key: str) -> str:
    if not extra_info:
        return ""
    return str(extra_info.get(key) or "")


def normalize_option(value: Any) -> str | None:
    return normalise_mme_answer(value)


def normalize_yes_no(value: Any) -> str | None:
    return normalise_yes_no(value)


def extract_mme_answer(output: str, extra_info: dict[str, Any] | None = None) -> str:
    option_a = _extra_value(extra_info, "option_a")
    option_b = _extra_value(extra_info, "option_b")
    return _extract_mme_answer(output, option_a=option_a, option_b=option_b) or ""


def mme_reward_fn(output: str, label: str) -> float:
    return float(score_mme_answer(output, label))

