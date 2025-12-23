"""
Robust OCM - A benchmark for optical compression models
"""

__version__ = "0.1.0"
__author__ = "Robust OCM Team"

from .render import TextRenderer
from .render.config import Config

__all__ = ["TextRenderer", "Config"]