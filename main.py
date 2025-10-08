from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon


class SubgroupNode:
    """A node in the segment discovery tree"""

    def __init__(self, data_indices, depth=0, conditions=None):
        self.data_indices = data_indices
        self.depth = depth
        self.conditions = conditions or []
        self.split_column = None
        self.split_value = None
        self.left = None
        self.right = None
        self.divergence = 0.0
        self.is_leaf = False

    def __repr__(self):
        cond_str = " AND ".join([f"{c[0]} {c[1]} {c[2]}" for c in self.conditions])
        return f"SubgroupNode(n={len(self.data_indices)}, div={self.divergence:.4f}, conditions=[{cond_str}])"


class SegmentationTree:
    """
    A decision tree that discovers segments with divergent metric distributions.
    """

    def __init__(
        self,
        max_depth=5,
        min_samples_split=50,
        min_samples_leaf=20,
        divergence_threshold=0.01,
        random_features=None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.divergence_threshold = divergence_threshold
        self.random_features = random_features  # Number of random features to consider (None = all)
        self.root = None
        self.background_dist = None
        self.metric_name = None
        self.is_continuous = None

    def _compute_divergence(self, metric_values):
        """Compute divergence from background distribution"""
        if len(metric_values) < 2:
            return 0.0

        if self.is_continuous:
            # Use Kolmogorov-Smirnov statistic for continuous distributions
            ks_stat, _ = stats.ks_2samp(metric_values, self.background_dist)
            return ks_stat
        else:
            # Use Jensen-Shannon divergence for discrete distributions
            # Create probability distributions
            segment_counts = Counter(metric_values)
            bg_counts = Counter(self.background_dist)

            # Get all unique values
            all_values = set(segment_counts.keys()) | set(bg_counts.keys())

            # Create probability vectors
            total_sub = len(metric_values)
            total_bg = len(self.background_dist)

            p = np.array([segment_counts.get(v, 0) / total_sub for v in all_values])
            q = np.array([bg_counts.get(v, 0) / total_bg for v in all_values])

            # Add small epsilon to avoid division by zero
            p = p + 1e-10
            q = q + 1e-10
            p = p / p.sum()
            q = q / q.sum()

            return jensenshannon(p, q)

    def _find_best_split(self, data, indices, available_columns):
        """Find the best column and value to split on"""
        best_divergence = 0
        best_column = None
        best_value = None
        best_left_indices = None
        best_right_indices = None

        # Sample random features if specified
        if self.random_features is not None:
            n_features = min(self.random_features, len(available_columns))
            columns_to_try = np.random.choice(available_columns, n_features, replace=False)
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
                left_metric = data.loc[left_indices, self.metric_name].values
                left_div = self._compute_divergence(left_metric)

                # We prioritize the split that creates maximum divergence
                if left_div > best_divergence:
                    best_divergence = left_div
                    best_column = col
                    best_value = val
                    best_left_indices = left_indices
                    best_right_indices = right_indices

        return best_column, best_value, best_left_indices, best_right_indices, best_divergence

    def _build_tree(self, data, indices, depth, conditions, available_columns):
        """Recursively build the tree"""
        node = SubgroupNode(indices, depth, conditions)

        # Compute divergence for this node
        metric_values = data.loc[indices, self.metric_name].values
        node.divergence = self._compute_divergence(metric_values)

        # Check stopping criteria (don't use divergence threshold for stopping, only for filtering)
        if depth >= self.max_depth or len(indices) < self.min_samples_split:
            node.is_leaf = True
            return node

        # Find best split
        col, val, left_idx, right_idx, divergence = self._find_best_split(
            data, indices, available_columns
        )

        if col is None:
            node.is_leaf = True
            return node

        # Create split
        node.split_column = col
        node.split_value = val

        # Build left subtree (where column == value) - this is our interesting segment
        left_conditions = conditions + [(col, "==", val)]
        node.left = self._build_tree(data, left_idx, depth + 1, left_conditions, available_columns)

        # Build right subtree (where column != value) - continue searching here too
        right_conditions = conditions + [(col, "!=", val)]
        node.right = self._build_tree(
            data, right_idx, depth + 1, right_conditions, available_columns
        )

        return node

    def fit(self, data, metric_column):
        """Fit the tree to discover segments"""
        self.metric_name = metric_column
        self.background_dist = data[metric_column].values

        # Determine if metric is continuous or discrete
        n_unique = data[metric_column].nunique()
        self.is_continuous = n_unique > 20  # Heuristic: >20 unique values = continuous

        available_columns = [col for col in data.columns if col != metric_column]
        indices = data.index.values

        self.root = self._build_tree(data, indices, 0, [], available_columns)

        return self

    def _collect_leaves(self, node, leaves):
        """Collect all leaf nodes"""
        if node is None:
            return

        if node.is_leaf and node.divergence > self.divergence_threshold:
            leaves.append(node)
        else:
            self._collect_leaves(node.left, leaves)
            self._collect_leaves(node.right, leaves)

    def get_segments(self, min_divergence=0.0):
        """Get all discovered segments"""
        leaves = []
        self._collect_leaves(self.root, leaves)

        # Filter and sort by divergence
        segments = [leaf for leaf in leaves if leaf.divergence >= min_divergence]
        segments.sort(key=lambda x: x.divergence, reverse=True)

        return segments


class SegmentationForest:
    """
    An ensemble of SegmentationTrees (Random Forest style)
    """

    def __init__(
        self,
        n_trees=10,
        max_depth=5,
        min_samples_split=50,
        min_samples_leaf=20,
        divergence_threshold=0.01,
        max_features=None,
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.divergence_threshold = divergence_threshold
        self.max_features = max_features  # Number of features to sample per tree
        self.trees = []
        self.metric_name = None

    def fit(self, data, metric_column):
        """Fit the forest"""
        self.metric_name = metric_column
        self.trees = []

        for _i in range(self.n_trees):
            # Create tree with random feature sampling
            tree = SegmentationTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                divergence_threshold=self.divergence_threshold,
                random_features=self.max_features,
            )

            # Bootstrap sample (Random Forest style)
            bootstrap_indices = np.random.choice(len(data), len(data), replace=True)
            bootstrap_data = data.iloc[bootstrap_indices].reset_index(drop=True)

            tree.fit(bootstrap_data, metric_column)
            self.trees.append(tree)

        return self

    def get_segments(self, min_support=2, min_divergence=0.0):
        """
        Get segments that appear in multiple trees
        min_support: minimum number of trees that must find this segment
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

        # Filter by support and compute average divergence
        robust_segments = []
        for conditions, segments in condition_groups.items():
            if len(segments) >= min_support:
                avg_divergence = np.mean([s.divergence for s in segments])
                avg_size = np.mean([len(s.data_indices) for s in segments])
                robust_segments.append(
                    {
                        "conditions": list(conditions),
                        "support": len(segments),
                        "avg_divergence": avg_divergence,
                        "avg_size": avg_size,
                    }
                )

        # Sort by support and divergence
        robust_segments.sort(key=lambda x: (x["support"], x["avg_divergence"]), reverse=True)

        return robust_segments


# Example usage with synthetic advertising data
def create_advertising_dataset(n_samples=10000):
    """Create synthetic advertising dataset with hidden segments"""
    np.random.seed(42)

    countries = np.random.choice(
        ["US", "UK", "CA", "DE", "FR"], n_samples, p=[0.4, 0.2, 0.15, 0.15, 0.1]
    )
    devices = np.random.choice(["Mobile", "Desktop", "Tablet"], n_samples, p=[0.5, 0.4, 0.1])
    genders = np.random.choice(["M", "F"], n_samples, p=[0.5, 0.5])
    time_of_day = np.random.choice(
        ["Morning", "Afternoon", "Evening", "Night"], n_samples, p=[0.2, 0.3, 0.3, 0.2]
    )

    # Base impressions: Poisson distribution
    impressions = np.random.poisson(100, n_samples)

    # Hidden patterns:
    # 1. Females in UK on Mobile have 3x impressions
    mask1 = (genders == "F") & (countries == "UK") & (devices == "Mobile")
    impressions[mask1] = np.random.poisson(300, mask1.sum())

    # 2. US users in the evening on Desktop have 2x impressions
    mask2 = (countries == "US") & (time_of_day == "Evening") & (devices == "Desktop")
    impressions[mask2] = np.random.poisson(200, mask2.sum())

    # 3. German users in the morning have very low impressions
    mask3 = (countries == "DE") & (time_of_day == "Morning")
    impressions[mask3] = np.random.poisson(30, mask3.sum())

    df = pd.DataFrame(
        {
            "country": countries,
            "device": devices,
            "gender": genders,
            "time_of_day": time_of_day,
            "impressions": impressions,
        }
    )

    return df


# Run the algorithm
print("Creating synthetic advertising dataset...")
data = create_advertising_dataset(10000)

print("\n=== Dataset Summary ===")
print(f"Shape: {data.shape}")
print("\nBackground distribution of impressions:")
print(f"Mean: {data['impressions'].mean():.2f}")
print(f"Median: {data['impressions'].median():.2f}")
print(f"Std: {data['impressions'].std():.2f}")

print("\n=== Running Single Tree ===")
tree = SegmentationTree(max_depth=3, min_samples_split=100, min_samples_leaf=50)
tree.fit(data, "impressions")
segments = tree.get_segments(min_divergence=0.05)

print(f"Found {len(segments)} interesting segments:\n")
for i, sg in enumerate(segments[:5], 1):
    conditions_str = " AND ".join([f"{c[0]} {c[1]} {c[2]}" for c in sg.conditions])
    metric_vals = data.loc[sg.data_indices, "impressions"]
    print(f"{i}. {conditions_str}")
    print(f"   Size: {len(sg.data_indices)} | Divergence: {sg.divergence:.4f}")
    print(
        f"   Mean impressions: {metric_vals.mean():.2f} (background: {data['impressions'].mean():.2f})"
    )
    print()

print("\n=== Running Random Forest (10 trees) ===")
forest = SegmentationForest(
    n_trees=10, max_depth=3, min_samples_split=100, min_samples_leaf=50, max_features=2
)
forest.fit(data, "impressions")
robust_segments = forest.get_segments(min_support=3, min_divergence=0.05)

print(f"Found {len(robust_segments)} robust segments:\n")
for i, sg in enumerate(robust_segments[:5], 1):
    conditions_str = " AND ".join([f"{c[0]} {c[1]} {c[2]}" for c in sg["conditions"]])
    print(f"{i}. {conditions_str}")
    print(
        f"   Support: {sg['support']}/{forest.n_trees} trees | Avg Divergence: {sg['avg_divergence']:.4f}"
    )
    print(f"   Avg Size: {sg['avg_size']:.0f}")
    print()

# Visualize the top segment
if robust_segments:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Get the top segment
    top_sg = robust_segments[0]

    # Filter data for this segment
    mask = np.ones(len(data), dtype=bool)
    for col, op, val in top_sg["conditions"]:
        if op == "==":
            mask &= data[col] == val
        elif op == "!=":
            mask &= data[col] != val

    segment_data = data[mask]["impressions"]

    # Plot distributions
    axes[0].hist(data["impressions"], bins=50, alpha=0.5, label="Background", density=True)
    axes[0].hist(segment_data, bins=50, alpha=0.5, label="Subgroup", density=True)
    axes[0].set_xlabel("Impressions")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Distribution Comparison")
    axes[0].legend()

    # Box plot
    plot_data = pd.DataFrame(
        {
            "Impressions": list(data["impressions"]) + list(segment_data),
            "Group": ["Background"] * len(data) + ["Subgroup"] * len(segment_data),
        }
    )
    sns.boxplot(data=plot_data, x="Group", y="Impressions", ax=axes[1])
    axes[1].set_title("Box Plot Comparison")

    plt.tight_layout()
    plt.savefig("segment_discovery_results.png", dpi=150, bbox_inches="tight")
    print("\nVisualization saved as 'segment_discovery_results.png'")
    plt.show()
