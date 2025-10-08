# Segmentation Forests

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

**Unsupervised segment discovery using divergence-based decision trees inspired by Random Forests.**

Automatically discover meaningful segments in your data where metric distributions significantly diverge from the background. Perfect for exploratory data analysis, anomaly detection, and customer segmentation without requiring labeled data.

---

## 🎯 The Problem

You have a dataset with many categorical features and a metric you care about. For example:

- **E-commerce**: Users across countries, devices, age groups → conversion rate
- **Digital advertising**: Impressions across demographics, platforms, times → CTR
- **Healthcare**: Patients across conditions, treatments, demographics → readmission rate
- **Finance**: Transactions across customer segments, times → fraud rate

**The question**: *Which specific combinations of features exhibit unusual behavior?*

With N features and many values per feature, exhaustively testing combinations is impossible. **Segmentation Forests solves this** by intelligently searching for segments where your metric's distribution differs significantly from the overall population.

---

## ✨ Key Features

- 🌳 **Tree-based discovery**: Greedy algorithm efficiently navigates combinatorial feature space
- 🌲 **Forest ensemble**: Bootstrap aggregating for robust, reproducible patterns
- 📊 **Statistical rigor**: KS distance (continuous) & Jensen-Shannon divergence (discrete)
- 📈 **Beautiful visualizations**: Distribution comparisons and quality assessments
- 🔬 **Fully typed**: Complete type hints for excellent IDE support
- ⚡ **Fast & scalable**: Handles datasets with millions of rows

---

## 🚀 Quick Start

### Installation

```bash
pip install segmentation-forests
```

### Basic Example

```python
import pandas as pd
from segmentation_forests import SegmentationTree, SegmentationForest

# Your data: features + metric
data = pd.DataFrame({
    'country': ['US', 'UK', 'US', 'UK', ...],
    'device': ['Mobile', 'Desktop', 'Mobile', ...],
    'gender': ['F', 'M', 'F', ...],
    'impressions': [245, 103, 312, 98, ...]  # Your metric
})

# Discover segments with a single tree
tree = SegmentationTree(max_depth=3, min_samples_split=100)
tree.fit(data, metric_column='impressions')
segments = tree.get_segments(min_divergence=0.1)

# View results
for i, seg in enumerate(segments[:3], 1):
    print(f"{i}. {seg.get_condition_string()}")
    print(f"   Divergence: {seg.divergence:.3f} | Size: {seg.size:,}")
```

**Output:**
```
1. gender == F AND device == Mobile AND country == UK
   Divergence: 0.948 | Size: 523

2. time_of_day == Evening AND country == US AND device == Desktop
   Divergence: 0.856 | Size: 412

3. country == DE AND time_of_day == Morning
   Divergence: 0.824 | Size: 289
```

---

## 🌲 Using the Forest (Recommended)

For more robust results, use the ensemble approach:

```python
from segmentation_forests import SegmentationForest

# Create forest with bootstrap sampling and random features
forest = SegmentationForest(
    n_trees=10,
    max_depth=3,
    max_features=2,  # Random feature selection
    min_samples_split=100,
    min_samples_leaf=50
)

# Fit and get segments found by multiple trees
forest.fit(data, metric_column='impressions')
robust_segments = forest.get_segments(min_support=3, min_divergence=0.1)

# View results
for seg in robust_segments:
    cond_str = " AND ".join([f"{c[0]} {c[1]} {c[2]}" for c in seg['conditions']])
    print(f"{cond_str}")
    print(f"  Support: {seg['support']}/10 trees ({seg['support_rate']*100:.0f}%)")
    print(f"  Avg Divergence: {seg['avg_divergence']:.3f}")
    print()
```

---

## 📊 Visualization

Beautiful distribution comparison plots:

```python
from segmentation_forests.visualization import plot_segment_comparison

# Compare segment distribution vs background
fig = plot_segment_comparison(
    data=data,
    segment_conditions=[('country', '==', 'UK'), ('device', '==', 'Mobile')],
    metric_column='impressions',
    title='UK Mobile Users vs Background'
)
fig.savefig('segment_comparison.png', dpi=150)
```

**Example output:**

The plot shows:
- **Left**: Overlapping histograms (background in blue, segment in coral)
- **Right**: Box plots comparing distributions
- **Clear separation**: Strong segments show minimal overlap

---

## 🧠 How It Works

### Algorithm Overview

1. **Compute Background Distribution**: Calculate the distribution of your metric across all data
2. **Greedy Tree Building**:
   - At each node, evaluate all feature-value splits
   - Choose the split that maximizes divergence from background
   - Recursively build left (matching condition) and right (not matching) subtrees
3. **Collect High-Divergence Leaves**: Return segments that diverge significantly
4. **Ensemble Aggregation** (Forest only): Vote across trees to find robust patterns

### Divergence Measures

The algorithm automatically selects the appropriate measure:

| Metric Type | Measure | Range | Description |
|------------|---------|-------|-------------|
| **Continuous** | Kolmogorov-Smirnov | [0, 1] | Max distance between CDFs |
| **Discrete** | Jensen-Shannon | [0, 1] | Symmetric KL divergence |

**Decision threshold**: ≤20 unique values → discrete, >20 → continuous

### Quality Guidelines

Interpret divergence scores:

- **≥ 0.5**: 🎯 **Excellent** - Strong, highly actionable pattern
- **0.3-0.5**: ✓ **Good** - Meaningful difference worth investigating
- **0.1-0.3**: ⚠️ **Weak** - Marginal effect, could be noise
- **< 0.1**: ❌ **Very weak** - Likely statistical noise

---

## 📖 API Reference

### `SegmentationTree`

```python
SegmentationTree(
    max_depth: int = 5,
    min_samples_split: int = 50,
    min_samples_leaf: int = 20,
    divergence_threshold: float = 0.01,
    random_features: Optional[int] = None
)
```

**Parameters:**
- `max_depth`: Maximum tree depth (controls segment complexity)
- `min_samples_split`: Minimum samples required to split a node
- `min_samples_leaf`: Minimum samples required in each child
- `divergence_threshold`: Minimum divergence to keep a segment
- `random_features`: Number of random features per split (None = use all)

**Methods:**
- `fit(data: pd.DataFrame, metric_column: str) -> Self`: Fit tree to data
- `get_segments(min_divergence: float = 0.0) -> List[SegmentationNode]`: Get segments

---

### `SegmentationForest`

```python
SegmentationForest(
    n_trees: int = 10,
    max_depth: int = 5,
    min_samples_split: int = 50,
    min_samples_leaf: int = 20,
    divergence_threshold: float = 0.01,
    max_features: Optional[int] = None
)
```

**Parameters:**
- `n_trees`: Number of trees in the forest
- Other parameters same as `SegmentationTree`

**Methods:**
- `fit(data: pd.DataFrame, metric_column: str) -> Self`: Fit forest
- `get_segments(min_support: int = 2, min_divergence: float = 0.0) -> List[Dict]`: Get robust segments

**Returns:** List of dicts with keys:
- `conditions`: List of (column, operator, value) tuples
- `support`: Number of trees that found this segment
- `avg_divergence`: Average divergence across trees
- `avg_size`: Average segment size
- `support_rate`: Fraction of trees (support / n_trees)

---

### `SegmentationNode`

Represents a discovered segment.

**Attributes:**
- `conditions`: List of (column, operator, value) tuples
- `divergence`: Divergence score
- `size`: Number of data points
- `depth`: Depth in tree
- `data_indices`: Indices of data points in this segment

**Methods:**
- `get_condition_string() -> str`: Human-readable condition string

---

## 🎨 Visualization Functions

### `plot_segment_comparison`

```python
plot_segment_comparison(
    data: pd.DataFrame,
    segment_conditions: List[Tuple],
    metric_column: str,
    title: Optional[str] = None,
    figsize: Tuple = (14, 5)
) -> plt.Figure
```

Creates side-by-side histogram and box plot comparison.

---

## 💡 Usage Tips

### Choosing Parameters

**For max_depth:**
- `depth=2`: Simple 2-condition segments (e.g., "Country=UK AND Device=Mobile")
- `depth=3-4`: **Recommended** - Balanced complexity
- `depth=5+`: Complex segments, risk of overfitting

**For min_divergence:**
- Start with `0.1` to see all interesting patterns
- Increase to `0.3+` to focus only on strong effects
- Use forest `min_support` to filter noise instead

**For forest:**
- `n_trees=10`: Good default
- `n_trees=20+`: More robust but slower
- `max_features=sqrt(n_features)`: Good for high-dimensional data

### Interpreting Results

1. **Always visualize top segments** to verify they make sense
2. **Check segment size** - very small segments may be spurious
3. **Use forest support** - patterns in 5+/10 trees are highly reliable
4. **Domain validation** - do discovered segments align with business intuition?

---

## 🔬 Example: Advertising Dataset

```python
from segmentation_forests import SegmentationForest
from segmentation_forests.visualization import plot_segment_comparison
import pandas as pd
import numpy as np

# Create synthetic advertising data
np.random.seed(42)
n = 10000

data = pd.DataFrame({
    'country': np.random.choice(['US', 'UK', 'CA', 'DE', 'FR'], n),
    'device': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n),
    'gender': np.random.choice(['M', 'F'], n),
    'time_of_day': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], n),
    'impressions': np.random.poisson(100, n)  # Base: ~100 impressions
})

# Add hidden pattern: UK females on mobile get 3x impressions
mask = (data['gender'] == 'F') & (data['country'] == 'UK') & (data['device'] == 'Mobile')
data.loc[mask, 'impressions'] = np.random.poisson(300, mask.sum())

# Discover the pattern
forest = SegmentationForest(n_trees=10, max_depth=3, max_features=2)
forest.fit(data, 'impressions')
segments = forest.get_segments(min_support=3, min_divergence=0.3)

# Result: Discovers the hidden pattern!
# Output: "gender == F AND country == UK AND device == Mobile"
# Divergence: 0.948, Support: 7/10 trees
```

See `examples/advertising_example.py` for the complete example.

---

## 🛠️ Development

### Setup

```bash
# Clone repository
git clone https://github.com/davidgeorgewilliams/segmentation-forests.git
cd segmentation-forests

# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# With coverage
pytest --cov=segmentation_forests --cov-report=html

# Run specific test
pytest tests/test_tree.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure all tests pass and code is formatted
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📚 Citation

If you use Segmentation Forests in your research or project, please cite:

```bibtex
@software{segmentation_forests,
  author = {Williams, David},
  title = {Segmentation Forests: Unsupervised Segment Discovery using Divergence-based Decision Trees},
  year = {2025},
  url = {https://github.com/davidgeorgewilliams/segmentation-forests}
}
```

---

## 🙏 Acknowledgments

- Algorithm inspired by Random Forests (Breiman, 2001)
- Divergence measures from information theory (Kullback-Leibler, Jensen-Shannon)
- Built with NumPy, pandas, SciPy, matplotlib, and seaborn

---

## 📞 Contact

**David Williams** - [david@davidgeorgewilliams.com](mailto:david@davidgeorgewilliams.com)

Project Link: [https://github.com/davidgeorgewilliams/segmentation-forests](https://github.com/davidgeorgewilliams/segmentation-forests)

---

**Happy Discovering! 🎯🌲**
