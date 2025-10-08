"""
Subgroup Discovery - Unsupervised algorithm for discovering interesting segments
in tabular data with divergent metric distributions.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

from .divergence import DivergenceMeasure, JensenShannonDistance, KSDistance
from .forest import SegmentationForest
from .node import SegmentationNode
from .tree import SegmentationTree

__all__ = [
    "SegmentationTree",
    "SegmentationForest",
    "SegmentationNode",
    "DivergenceMeasure",
    "KSDistance",
    "JensenShannonDistance",
]
