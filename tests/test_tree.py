"""
Unit tests for SegmentationTree
"""

import numpy as np
import pandas as pd
import pytest

from segmentation_forests import SegmentationForest, SegmentationTree
from segmentation_forests.divergence import JensenShannonDistance, KSDistance
from segmentation_forests.node import SegmentationNode


def create_test_data():
    """Create simple test dataset."""
    np.random.seed(42)
    n = 1000

    data = pd.DataFrame(
        {
            "cat_a": np.random.choice(["A", "B"], n),
            "cat_b": np.random.choice(["X", "Y", "Z"], n),
            "metric": np.random.poisson(10, n),
        }
    )

    # Add a clear pattern
    mask = (data["cat_a"] == "A") & (data["cat_b"] == "X")
    data.loc[mask, "metric"] = np.random.poisson(30, mask.sum())

    return data


class TestSegmentationNode:
    def test_node_creation(self):
        indices = np.array([0, 1, 2, 3, 4])
        node = SegmentationNode(indices, depth=0)

        assert node.size == 5
        assert node.depth == 0
        assert not node.is_leaf
        assert len(node.conditions) == 0

    def test_node_with_conditions(self):
        indices = np.array([0, 1, 2])
        conditions = [("cat_a", "==", "A"), ("cat_b", "==", "X")]
        node = SegmentationNode(indices, depth=2, conditions=conditions)

        assert node.depth == 2
        assert len(node.conditions) == 2
        assert "cat_a == A" in node.get_condition_string()


class TestDivergence:
    def test_ks_distance(self):
        measure = KSDistance()

        # Same distribution
        x = np.random.normal(0, 1, 100)
        y = np.random.normal(0, 1, 100)
        div = measure.compute(x, y)
        assert 0 <= div <= 1

        # Different distributions
        x = np.random.normal(0, 1, 100)
        y = np.random.normal(5, 1, 100)
        div = measure.compute(x, y)
        assert div > 0.5

    def test_js_divergence(self):
        measure = JensenShannonDistance()

        # Same distribution
        x = np.random.choice([1, 2, 3], 100)
        y = np.random.choice([1, 2, 3], 100)
        div = measure.compute(x, y)
        assert 0 <= div <= 1


class TestSegmentationTree:
    def test_tree_fit(self):
        data = create_test_data()
        tree = SegmentationTree(max_depth=2, min_samples_split=50)

        tree.fit(data, "metric")

        assert tree.root is not None
        assert tree.metric_name == "metric"
        assert tree.background_dist is not None

    def test_get_segments(self):
        data = create_test_data()
        tree = SegmentationTree(max_depth=3, min_samples_split=50)
        tree.fit(data, "metric")

        segments = tree.get_segments(min_divergence=0.0)

        assert len(segments) > 0
        assert all(isinstance(sg, SegmentationNode) for sg in segments)
        # Should be sorted by divergence
        assert all(
            segments[i].divergence >= segments[i + 1].divergence for i in range(len(segments) - 1)
        )

    def test_finds_planted_pattern(self):
        data = create_test_data()
        tree = SegmentationTree(max_depth=3, min_samples_split=30)
        tree.fit(data, "metric")

        segments = tree.get_segments(min_divergence=0.1)

        # Should find the A+X pattern
        found_pattern = False
        for sg in segments:
            conditions_dict = {c[0]: c[2] for c in sg.conditions if c[1] == "=="}
            if conditions_dict.get("cat_a") == "A" and conditions_dict.get("cat_b") == "X":
                found_pattern = True
                break

        assert found_pattern, "Tree should find the planted pattern"


class TestSegmentationForest:
    def test_forest_fit(self):
        data = create_test_data()
        forest = SegmentationForest(n_trees=5, max_depth=2)

        forest.fit(data, "metric")

        assert len(forest.trees) == 5
        assert all(tree.root is not None for tree in forest.trees)

    def test_get_robust_segments(self):
        data = create_test_data()
        forest = SegmentationForest(n_trees=5, max_depth=3, max_features=1)
        forest.fit(data, "metric")

        segments = forest.get_segments(min_support=2, min_divergence=0.0)

        assert len(segments) > 0
        assert all(sg["support"] >= 2 for sg in segments)
        assert all("avg_divergence" in sg for sg in segments)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
