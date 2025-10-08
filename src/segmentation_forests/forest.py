"""Random forest ensemble for robust segment discovery."""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .tree import SegmentationTree


class SegmentationForest:
    """An ensemble of SegmentationTrees (Random Forest style).

    Creates multiple trees with bootstrap sampling and random feature selection
    to discover robust segments that appear consistently across trees.

    Attributes:
        n_trees: Number of trees in the forest
        max_depth: Maximum depth of each tree
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required in a leaf node
        divergence_threshold: Minimum divergence for interesting segments
        max_features: Number of features to sample per tree split
    """

    def __init__(
        self,
        n_trees: int = 10,
        max_depth: int = 5,
        min_samples_split: int = 50,
        min_samples_leaf: int = 20,
        divergence_threshold: float = 0.01,
        max_features: Optional[int] = None,
    ):
        """Initialize SegmentationForest.

        Args:
            n_trees: Number of trees in the forest
            max_depth: Maximum depth of each tree
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required in a leaf node
            divergence_threshold: Minimum divergence for interesting segments
            max_features: Number of features to sample per tree split
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.divergence_threshold = divergence_threshold
        self.max_features = max_features

        self.trees: List[SegmentationTree] = []
        self.metric_name: Optional[str] = None

    def fit(self, data: pd.DataFrame, metric_column: str) -> "SegmentationForest":
        """Fit the forest.

        Args:
            data: DataFrame containing features and metric
            metric_column: Name of the metric column

        Returns:
            Self for method chaining
        """
        self.metric_name = metric_column
        self.trees = []

        for _ in range(self.n_trees):
            # Create tree with random feature sampling
            tree = SegmentationTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                divergence_threshold=self.divergence_threshold,
                random_features=self.max_features,
            )

            # Bootstrap sample
            bootstrap_indices = np.random.choice(len(data), len(data), replace=True)
            bootstrap_data = data.iloc[bootstrap_indices].reset_index(drop=True)

            tree.fit(bootstrap_data, metric_column)
            self.trees.append(tree)

        return self

    def get_segments(self, min_support: int = 2, min_divergence: float = 0.0) -> List[Dict]:
        """Get segments that appear in multiple trees.

        Args:
            min_support: Minimum number of trees that must find this segment
            min_divergence: Minimum average divergence

        Returns:
            List of dictionaries containing segment information
        """
        # Collect all segments from all trees
        all_segments = []
        for tree in self.trees:
            segments = tree.get_segments(min_divergence)
            all_segments.extend(segments)

        # Group by conditions
        condition_groups = defaultdict(list)
        for segment in all_segments:
            # Create a hashable key from conditions
            key = frozenset(segment.conditions)
            condition_groups[key].append(segment)

        # Filter by support and compute statistics
        robust_segments = []
        for conditions, segments in condition_groups.items():
            if len(segments) >= min_support:
                avg_divergence = np.mean([s.divergence for s in segments])
                avg_size = np.mean([s.size for s in segments])
                robust_segments.append(
                    {
                        "conditions": list(conditions),
                        "support": len(segments),
                        "avg_divergence": avg_divergence,
                        "avg_size": avg_size,
                        "support_rate": len(segments) / self.n_trees,
                    }
                )

        # Sort by support and divergence
        robust_segments.sort(key=lambda x: (x["support"], x["avg_divergence"]), reverse=True)

        return robust_segments
