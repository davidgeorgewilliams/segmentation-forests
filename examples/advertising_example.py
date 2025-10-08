"""
Example: Discovering segments in advertising data
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from segmentation_forests import SegmentationForest, SegmentationTree
from segmentation_forests.visualization import plot_segment_comparison


def create_advertising_dataset(n_samples=10000, seed=42):
    """Create synthetic advertising dataset with hidden patterns."""
    np.random.seed(seed)

    # Create features
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


def main():
    print("=" * 70)
    print("Segmentation Forests: Advertising Example")
    print("=" * 70)

    # Create dataset
    print("\n📊 Creating synthetic advertising dataset...")
    data = create_advertising_dataset(n_samples=10000)

    print(f"Dataset shape: {data.shape}")
    print(f"\nFeatures: {list(data.columns[:-1])}")
    print("Metric: impressions")
    print("\nBackground distribution:")
    print(f"  Mean: {data['impressions'].mean():.2f}")
    print(f"  Median: {data['impressions'].median():.2f}")
    print(f"  Std: {data['impressions'].std():.2f}")

    # Single tree
    print("\n" + "=" * 70)
    print("🌳 Single Tree Discovery")
    print("=" * 70)

    tree = SegmentationTree(
        max_depth=3, min_samples_split=100, min_samples_leaf=50, divergence_threshold=0.05
    )

    tree.fit(data, "impressions")
    segments = tree.get_segments(min_divergence=0.1)

    print(f"\nFound {len(segments)} interesting segments:\n")
    for i, sg in enumerate(segments[:5], 1):
        metric_vals = data.loc[sg.data_indices, "impressions"]
        print(f"{i}. {sg.get_condition_string()}")
        print(f"   Size: {sg.size:,} | Divergence: {sg.divergence:.4f}")
        print(f"   Mean: {metric_vals.mean():.2f} (vs {data['impressions'].mean():.2f})")
        print()

    # Forest
    print("=" * 70)
    print("🌲 Random Forest Discovery (More Robust)")
    print("=" * 70)

    forest = SegmentationForest(
        n_trees=10,
        max_depth=3,
        min_samples_split=100,
        min_samples_leaf=50,
        divergence_threshold=0.05,
        max_features=2,
    )

    forest.fit(data, "impressions")
    robust_segments = forest.get_segments(min_support=3, min_divergence=0.1)

    print(f"\nFound {len(robust_segments)} robust segments:\n")

    # Separate into highly interesting vs marginal segments
    high_quality = [sg for sg in robust_segments if sg["avg_divergence"] >= 0.3]
    marginal = [sg for sg in robust_segments if sg["avg_divergence"] < 0.3]

    if high_quality:
        print("🎯 HIGH QUALITY SEGMENTS (divergence ≥ 0.3):")
        for i, sg in enumerate(high_quality, 1):
            cond_str = " AND ".join([f"{c[0]} {c[1]} {c[2]}" for c in sg["conditions"]])
            print(f"\n{i}. {cond_str}")
            print(
                f"   Support: {sg['support']}/{forest.n_trees} trees "
                f"({sg['support_rate']*100:.0f}%)"
            )
            print(f"   Avg Divergence: {sg['avg_divergence']:.4f}")
            print(f"   Avg Size: {sg['avg_size']:.0f}")

    if marginal:
        print("\n⚠️  MARGINAL SEGMENTS (divergence < 0.3) - May be noise:")
        for i, sg in enumerate(marginal[:3], 1):
            cond_str = " AND ".join([f"{c[0]} {c[1]} {c[2]}" for c in sg["conditions"]])
            if len(cond_str) > 50:
                cond_str = cond_str[:47] + "..."
            print(f"   {i}. {cond_str} (div: {sg['avg_divergence']:.4f})")
        if len(marginal) > 3:
            print(f"   ... and {len(marginal) - 3} more")

    # Visualizations - plot all segments above threshold
    if robust_segments:
        print("\n" + "=" * 70)
        print("📈 Creating Visualizations")
        print("=" * 70)

        # Create plots directory
        plots_dir = Path("examples/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📁 Saving plots to: {plots_dir}/")

        # Create individual visualizations for all segments
        num_to_plot = len(robust_segments)
        print(f"\n📊 Creating distribution comparison plots for {num_to_plot} segments...")

        for i, sg in enumerate(robust_segments, 1):
            cond_str = " AND ".join(
                [f"{c[0]}={c[2]}" if c[1] == "==" else f"{c[0]}≠{c[2]}" for c in sg["conditions"]]
            )
            # Truncate long titles
            display_cond = cond_str
            if len(display_cond) > 60:
                display_cond = display_cond[:57] + "..."

            fig = plot_segment_comparison(
                data,
                sg["conditions"],
                "impressions",
                title=f"Segment {i}: {display_cond} (div={sg['avg_divergence']:.3f})",
            )
            output_path = plots_dir / f"segment_{i}_comparison.png"
            fig.savefig(output_path, dpi=150, bbox_inches="tight")

            # Show quality indicator
            quality = (
                "🎯 STRONG"
                if sg["avg_divergence"] >= 0.5
                else "✓ GOOD" if sg["avg_divergence"] >= 0.3 else "⚠ WEAK"
            )
            print(
                f"   {quality} | Saved: segment_{i}_comparison.png (div={sg['avg_divergence']:.3f})"
            )
            plt.close(fig)

        print("\n💡 Quality Guide: 🎯 ≥0.5 (strong) | ✓ ≥0.3 (good) | ⚠ <0.3 (weak)")
    else:
        print("\n⚠️  No segments found above the divergence threshold")
        print("    Try lowering min_divergence or adjusting other parameters")

    print("\n" + "=" * 70)
    print("✨ Done! Check the plots in examples/plots/")
    print("=" * 70)
    print("\n📊 Quality Guidelines:")
    print("   • Divergence ≥ 0.5: Excellent - strong, actionable pattern")
    print("   • Divergence 0.3-0.5: Good - meaningful difference")
    print("   • Divergence 0.1-0.3: Weak - often noise, investigate carefully")
    print("   • Divergence < 0.1: Very weak - likely statistical noise")
    print("\n💡 Tips for better results:")
    print("   • Increase min_divergence threshold to filter out noise")
    print("   • Increase min_support to require more tree agreement")
    print("   • Adjust max_depth to find more/fewer conditions per segment")
    print("   • Try more trees (n_trees=20+) for more robust results")


if __name__ == "__main__":
    main()
