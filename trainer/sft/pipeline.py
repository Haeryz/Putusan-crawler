"""Backward-compatible import for the explicit workflow in main.py."""

from .main import run_training

__all__ = ["run_training"]
