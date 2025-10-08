"""Decision tree for segment discovery."""

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from .divergence import DivergenceMeasure, create_divergence_measure
from .node import SegmentationNode


class SegmentationTree:
    """A decision tree that discovers segments with divergent metric distributions.

    This tree uses a greedy algorithm to find segments where the metric
    distribution significantly differs from the background distribution.

    Attributes:
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required in a leaf node
        divergence_threshold: Minimum divergence to consider a segment interesting
        random_features: Number of random features to consider (None = all)
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 50,
        min_samples_leaf: int = 20,
        divergence_threshold: float = 0.01,
        random_features: Optional[int] = None,
    ):
        """Initialize SegmentationTree.

        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required in a leaf node
            divergence_threshold: Minimum divergence for interesting segments
            random_features: Number of random features to consider per split
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.divergence_threshold = divergence_threshold
        self.random_features = random_features

        self.root: Optional[SegmentationNode] = None
        self.background_dist: Optional[np.ndarray] = None
        self.metric_name: Optional[str] = None
        self.divergence_measure: Optional[DivergenceMeasure] = None

    def _compute_divergence(self, metric_values: np.ndarray) -> float:
        """Compute divergence from background distribution."""
        assert self.divergence_measure is not None, "Must fit tree before computing divergence"
        assert self.background_dist is not None, "Must fit tree before computing divergence"
        return self.divergence_measure.compute(metric_values, self.background_dist)

    def _find_best_split(
        self, data: pd.DataFrame, indices: np.ndarray, available_columns: List[str]
    ) -> Tuple[Optional[str], Optional[Any], Optional[np.ndarray], Optional[np.ndarray], float]:
        """Find the best column and value to split on.

        Returns:
            Tuple of (column, value, left_indices, right_indices, divergence)
        """
        best_divergence: float = 0.0
        best_column: Optional[str] = None
        best_value: Optional[Any] = None
        best_left_indices: Optional[np.ndarray] = None
        best_right_indices: Optional[np.ndarray] = None

        # Sample random features if specified
        columns_to_try: List[str]
        if self.random_features is not None:
            n_features = min(self.random_features, len(available_columns))
            columns_to_try = list(np.random.choice(available_columns, n_features, replace=False))
        else:
            columns_to_try = available_columns

        for col in columns_to_try:
            if col == self.metric_name:
                continue

            # Get unique values for this column
            unique_values = data.loc[indices, col].unique()

            # Try each unique value as a split point
            for val in unique_values:
                # Create split
                left_mask = data.loc[indices, col] == val
                left_indices = indices[left_mask]
                right_indices = indices[~left_mask]

                # Check minimum samples constraint
                if (
                    len(left_indices) < self.min_samples_leaf
                    or len(right_indices) < self.min_samples_leaf
                ):
                    continue

                # Compute divergence for left split
                assert self.metric_name is not None
                left_metric = data.loc[left_indices, self.metric_name].values
                left_div = self._compute_divergence(left_metric)

                # Keep track of best split
                if left_div > best_divergence:
                    best_divergence = left_div
                    best_column = col
                    best_value = val
                    best_left_indices = left_indices
                    best_right_indices = right_indices

        return best_column, best_value, best_left_indices, best_right_indices, best_divergence

    def _build_tree(
        self,
        data: pd.DataFrame,
        indices: np.ndarray,
        depth: int,
        conditions: List[Tuple[str, str, Any]],
        available_columns: List[str],
    ) -> SegmentationNode:
        """Recursively build the tree."""
        node = SegmentationNode(indices, depth, conditions)

        # Compute divergence for this node
        assert self.metric_name is not None
        metric_values = data.loc[indices, self.metric_name].values
        node.divergence = self._compute_divergence(metric_values)

        # Check stopping criteria
        if depth >= self.max_depth or len(indices) < self.min_samples_split:
            node.is_leaf = True
            return node

        # Find best split
        col, val, left_idx, right_idx, divergence = self._find_best_split(
            data, indices, available_columns
        )

        if col is None or left_idx is None or right_idx is None:
            node.is_leaf = True
            return node

        # Create split
        node.split_column = col
        node.split_value = val

        # Build left subtree (where column == value)
        left_conditions = conditions + [(col, "==", val)]
        node.left = self._build_tree(data, left_idx, depth + 1, left_conditions, available_columns)

        # Build right subtree (where column != value)
        right_conditions = conditions + [(col, "!=", val)]
        node.right = self._build_tree(
            data, right_idx, depth + 1, right_conditions, available_columns
        )

        return node

    def fit(self, data: pd.DataFrame, metric_column: str) -> "SegmentationTree":
        """Fit the tree to discover segments.

        Args:
            data: DataFrame containing features and metric
            metric_column: Name of the metric column

        Returns:
            Self for method chaining
        """
        self.metric_name = metric_column
        self.background_dist = data[metric_column].values

        # Create appropriate divergence measure
        self.divergence_measure = create_divergence_measure(self.background_dist)

        available_columns = [col for col in data.columns if col != metric_column]
        indices = data.index.values

        self.root = self._build_tree(data, indices, 0, [], available_columns)

        return self

    def _collect_leaves(
        self, node: Optional[SegmentationNode], leaves: List[SegmentationNode]
    ) -> None:
        """Collect all leaf nodes recursively."""
        if node is None:
            return

        if node.is_leaf and node.divergence > self.divergence_threshold:
            leaves.append(node)
        else:
            self._collect_leaves(node.left, leaves)
            self._collect_leaves(node.right, leaves)

    def get_segments(self, min_divergence: float = 0.0) -> List[SegmentationNode]:
        """Get all discovered segments sorted by divergence.

        Args:
            min_divergence: Minimum divergence threshold

        Returns:
            List of SegmentationNode objects sorted by divergence
        """
        leaves: List[SegmentationNode] = []
        self._collect_leaves(self.root, leaves)

        # Filter and sort by divergence
        segments = [leaf for leaf in leaves if leaf.divergence >= min_divergence]
        segments.sort(key=lambda x: x.divergence, reverse=True)

        return segments
