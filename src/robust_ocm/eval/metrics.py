# -*- coding: utf-8 -*-
"""
Evaluation metrics for OCR predictions.

This module provides classes for calculating various evaluation metrics
including Character Error Rate (CER) and BLEU scores.
"""

import logging
import re
from typing import Dict, Any, List, Tuple
import Levenshtein as lev
from sacrebleu import BLEU
from tqdm import tqdm

from .preprocessing import normalize_text, normalize_formula, normalize_html_table

logger = logging.getLogger(__name__)


class CharacterErrorRate:
    """Character Error Rate metric."""

    def __init__(self, normalize: bool = True):
        """
        Initialize CER metric.

        Args:
            normalize: Whether to normalize text before calculating CER
        """
        self.normalize = normalize

    def calculate(self, pred: str, gt: str) -> float:
        """
        Calculate CER between prediction and ground truth.

        Args:
            pred: Predicted text
            gt: Ground truth text

        Returns:
            CER value (0-100)
        """
        if self.normalize:
            pred = normalize_text(pred)
            gt = normalize_text(gt)

        if not gt:
            return 100.0 if pred else 0.0

        distance = lev.distance(gt, pred)
        cer = distance / max(len(gt), len(pred)) * 100
        return cer


class BLEUScore:
    """BLEU metric for text similarity."""

    def __init__(self, normalize: bool = True):
        """
        Initialize BLEU metric.

        Args:
            normalize: Whether to normalize text before calculating BLEU
        """
        self.normalize = normalize
        self.bleu = BLEU(effective_order=True)

    def calculate(self, pred: str, gt: str) -> float:
        """
        Calculate BLEU score between prediction and ground truth.

        Args:
            pred: Predicted text
            gt: Ground truth text

        Returns:
            BLEU score (0-100)
        """
        if self.normalize:
            pred = normalize_text(pred)
            gt = normalize_text(gt)

        if not gt:
            return 0.0

        # Calculate BLEU (sacrebleu expects strings)
        score = self.bleu.corpus_score([pred], [[gt]])
        return score.score


def calculate_metrics(pred: str, gt: str, normalize: bool = True) -> Dict[str, float]:
    """
    Calculate all metrics for a single prediction-ground truth pair.

    Args:
        pred: Predicted text
        gt: Ground truth text
        normalize: Whether to normalize text before calculating metrics

    Returns:
        Dictionary of metric scores
    """
    cer = CharacterErrorRate(normalize=normalize)
    bleu = BLEUScore(normalize=normalize)

    return {
        "cer": cer.calculate(pred, gt),
        "bleu": bleu.calculate(pred, gt),
    }


def calculate_all_metrics(
    predictions: Dict[str, str],
    ground_truths: Dict[str, str],
    normalize: bool = True,
) -> Dict[str, Any]:
    """
    Calculate metrics for all predictions and aggregate results.

    Args:
        predictions: Dictionary mapping page_id to predicted text
        ground_truths: Dictionary mapping page_id to ground truth text
        normalize: Whether to normalize text before calculating metrics

    Returns:
        Dictionary containing aggregated metrics and per-page results
    """
    results = {
        "metrics": {"cer": [], "bleu": []},
        "per_page": {},
        "summary": {},
    }

    for page_id, pred_text in tqdm(predictions.items(), desc="Calculating metrics", unit="page"):
        if page_id not in ground_truths:
            continue

        gt_text = ground_truths[page_id]
        metrics = calculate_metrics(pred_text, gt_text, normalize=normalize)

        results["per_page"][page_id] = metrics
        for metric_name, value in metrics.items():
            results["metrics"][metric_name].append(value)

    # Calculate summary statistics
    for metric_name, values in results["metrics"].items():
        if values:
            results["summary"][f"{metric_name}_mean"] = sum(values) / len(values)
            results["summary"][f"{metric_name}_std"] = (
                sum((x - results["summary"][f"{metric_name}_mean"]) ** 2 for x in values)
                / len(values)
            ) ** 0.5
            results["summary"][f"{metric_name}_min"] = min(values)
            results["summary"][f"{metric_name}_max"] = max(values)
            results["summary"][f"{metric_name}_count"] = len(values)

    return results