"""Node representation for segment discovery trees."""

from typing import Any, List, Optional, Tuple

import numpy as np


class SegmentationNode:
    """A node in the segment discovery tree.

    Attributes:
        data_indices: Indices of data points in this node
        depth: Depth of this node in the tree
        conditions: List of (column, operator, value) tuples defining this segment
        split_column: Column used for splitting (None if leaf)
        split_value: Value used for splitting (None if leaf)
        left: Left child node
        right: Right child node
        divergence: Divergence score from background distribution
        is_leaf: Whether this is a leaf node
    """

    def __init__(
        self,
        data_indices: np.ndarray,
        depth: int = 0,
        conditions: Optional[List[Tuple[str, str, Any]]] = None,
    ):
        """Initialize a SegmentationNode.

        Args:
            data_indices: Array of data point indices in this node
            depth: Depth of this node in the tree
            conditions: List of conditions defining this segment
        """
        self.data_indices = data_indices
        self.depth = depth
        self.conditions = conditions or []
        self.split_column: Optional[str] = None
        self.split_value: Optional[Any] = None
        self.left: Optional["SegmentationNode"] = None
        self.right: Optional["SegmentationNode"] = None
        self.divergence: float = 0.0
        self.is_leaf: bool = False

    def __repr__(self) -> str:
        """String representation of the node."""
        cond_str = " AND ".join([f"{c[0]} {c[1]} {c[2]}" for c in self.conditions])
        return (
            f"SegmentationNode(n={len(self.data_indices)}, "
            f"div={self.divergence:.4f}, "
            f"conditions=[{cond_str}])"
        )

    @property
    def size(self) -> int:
        """Number of data points in this node."""
        return len(self.data_indices)

    def get_condition_string(self) -> str:
        """Get human-readable condition string."""
        if not self.conditions:
            return "All data"
        return " AND ".join([f"{col} {op} {val}" for col, op, val in self.conditions])
