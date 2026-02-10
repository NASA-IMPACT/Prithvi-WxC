"""Prithvi-WxC - Weather and climate foundational model."""

__version__ = "1.0.0"

from . import dataloaders, model, download

__all__ = [
    "dataloaders",
    "model",
    "download",
]
