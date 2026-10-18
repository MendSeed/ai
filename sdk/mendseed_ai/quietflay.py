"""
Backward compatibility shim for quietflay module.
Canonical implementation is now in mendseed_ai.quietfray.
"""

from .quietfray import QuietFrayEngine, QuietFlayEngine

__all__ = ["QuietFrayEngine", "QuietFlayEngine"]
