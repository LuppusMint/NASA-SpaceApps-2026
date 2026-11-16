"""Self-contained scientific core for Fire Relay South America.

Sub-modules
-----------
- :mod:`src.core.geogrid`      equal-area gridding and boundary helpers
- :mod:`src.core.splits_st`    frozen spatial-block and temporal splits
- :mod:`src.core.harmonize`    robust bridges, Kalman and hierarchical reconciliation
- :mod:`src.core.postprocess`  fold-wise sigma and hurdle calibration
- :mod:`src.core.stacking`     simplex ensemble optimisation
- :mod:`src.core.trends`       Theil-Sen, Hamed-Rao Mann-Kendall and BH FDR
- :mod:`src.core.extremes`     block maxima and GEV return levels
- :mod:`src.core.metrics`      benchmark metric implementations
- :mod:`src.core.duckdb_helper` portable DuckDB connection settings
"""

from __future__ import annotations

__all__ = [
    "duckdb_helper",
    "metrics",
    "geogrid",
    "splits_st",
    "harmonize",
    "postprocess",
    "stacking",
    "trends",
    "extremes",
]
