"""Visualization utilities for segment discovery results."""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_segment_comparison(
    data: pd.DataFrame,
    segment_conditions: List[tuple],
    metric_column: str,
    title: Optional[str] = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Plot comparison between background and segment distributions.

    Args:
        data: Full dataset
        segment_conditions: List of (column, operator, value) tuples
        metric_column: Name of the metric column
        title: Optional plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    # Filter data for segment
    mask = np.ones(len(data), dtype=bool)
    for col, op, val in segment_conditions:
        if op == "==":
            mask &= data[col] == val
        elif op == "!=":
            mask &= data[col] != val

    segment_data = data[mask][metric_column]
    background_data = data[metric_column]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram comparison
    axes[0].hist(
        background_data, bins=50, alpha=0.5, label="Background", density=True, color="skyblue"
    )
    axes[0].hist(segment_data, bins=50, alpha=0.5, label="Segment", density=True, color="coral")
    axes[0].set_xlabel(metric_column.replace("_", " ").title())
    axes[0].set_ylabel("Density")
    axes[0].set_title("Distribution Comparison")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Box plot comparison
    plot_df = pd.DataFrame(
        {
            metric_column: list(background_data) + list(segment_data),
            "Group": ["Background"] * len(background_data) + ["Segment"] * len(segment_data),
        }
    )
    sns.boxplot(data=plot_df, x="Group", y=metric_column, ax=axes[1])
    axes[1].set_title("Box Plot Comparison")
    axes[1].grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()
    return fig
