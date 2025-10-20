"""
Evaluation utilities for LoCoBench
"""

try:
    from .evaluator import LoCoBenchEvaluator
    __all__ = [
        "LoCoBenchEvaluator"
    ]
except Exception:  # pragma: no cover - optional import
    # If evaluator requires heavy optional deps, allow package to be imported without it.
    __all__ = []