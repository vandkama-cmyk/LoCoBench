"""
LoCoBench: A Novel Benchmark for Evaluating Long-Context Language Models 
in Complex Software Development Tasks

This package provides tools for generating, evaluating, and analyzing
long-context LLM performance on large-scale codebases.
"""

__version__ = "0.1.0"
__author__ = "LoCoBench Team"

from .core import *
from .analysis import *
from .evaluation import *

# generation imports may require heavy optional dependencies (openai, google genai, etc.).
# Import lazily and fail gracefully so light-weight utilities (e.g. utils.rag) can be used
# without installing all provider SDKs.
try:
    from .generation import *
except Exception as e:  # pragma: no cover - best-effort graceful import
    import warnings
    warnings.warn(f"Could not import locobench.generation (optional): {e}")

__all__ = [
    "__version__",
    "__author__",
] 