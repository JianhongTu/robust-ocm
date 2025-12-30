# -*- coding: utf-8 -*-
"""
Evaluation module for OCR predictions.

This module provides tools for evaluating OCR predictions against ground truth
using metrics from the OmniDocBench benchmark.
"""

from .metrics import (
    calculate_metrics,
    calculate_all_metrics,
    CharacterErrorRate,
    BLEUScore,
)
from .preprocessing import (
    normalize_text,
    normalize_formula,
    normalize_html_table,
    normalize_latex_table,
    clean_string,
    textblock_to_unicode,
    remove_markdown_fences,
)
from .cli import main

__all__ = [
    "calculate_metrics",
    "calculate_all_metrics",
    "CharacterErrorRate",
    "BLEUScore",
    "normalize_text",
    "normalize_formula",
    "normalize_html_table",
    "normalize_latex_table",
    "clean_string",
    "textblock_to_unicode",
    "remove_markdown_fences",
    "main",
]