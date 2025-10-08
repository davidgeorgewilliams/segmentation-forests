"""Divergence measures for comparing distributions."""

from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon


class DivergenceMeasure(ABC):
    """Abstract base class for divergence measures."""

    @abstractmethod
    def compute(self, segment_values: np.ndarray, background_values: np.ndarray) -> float:
        """Compute divergence between segment and background distributions.

        Args:
            segment_values: Values from the segment
            background_values: Values from the background distribution

        Returns:
            Divergence score (higher = more different)
        """
        pass


class KSDistance(DivergenceMeasure):
    """Kolmogorov-Smirnov distance for continuous distributions."""

    def compute(self, segment_values: np.ndarray, background_values: np.ndarray) -> float:
        """Compute KS statistic between two continuous distributions.

        The KS statistic measures the maximum distance between the
        cumulative distribution functions of two samples.

        Args:
            segment_values: Values from the segment
            background_values: Values from the background distribution

        Returns:
            KS statistic in [0, 1]
        """
        if len(segment_values) < 2:
            return 0.0

        ks_stat, _ = stats.ks_2samp(segment_values, background_values)
        return float(ks_stat)


class JensenShannonDistance(DivergenceMeasure):
    """Jensen-Shannon divergence for discrete distributions."""

    def compute(self, segment_values: np.ndarray, background_values: np.ndarray) -> float:
        """Compute Jensen-Shannon divergence between two discrete distributions.

        JS divergence is a symmetric version of KL divergence, bounded in [0, 1].

        Args:
            segment_values: Values from the segment
            background_values: Values from the background distribution

        Returns:
            JS divergence in [0, 1]
        """
        if len(segment_values) < 2:
            return 0.0

        # Create probability distributions
        segment_counts = Counter(segment_values)
        bg_counts = Counter(background_values)

        # Get all unique values
        all_values = sorted(set(segment_counts.keys()) | set(bg_counts.keys()))

        # Create probability vectors
        total_sub = len(segment_values)
        total_bg = len(background_values)

        p = np.array([segment_counts.get(v, 0) / total_sub for v in all_values])
        q = np.array([bg_counts.get(v, 0) / total_bg for v in all_values])

        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        p = p / p.sum()
        q = q / q.sum()

        return float(jensenshannon(p, q))


def create_divergence_measure(metric_values: np.ndarray, threshold: int = 20) -> DivergenceMeasure:
    """Factory function to create appropriate divergence measure.

    Args:
        metric_values: Sample of metric values
        threshold: Number of unique values above which we consider continuous

    Returns:
        Appropriate divergence measure instance
    """
    n_unique = len(np.unique(metric_values))

    if n_unique > threshold:
        return KSDistance()
    else:
        return JensenShannonDistance()
