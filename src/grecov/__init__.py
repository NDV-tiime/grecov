"""grecov — Exact confidence intervals for multinomial distributions."""

__version__ = "0.1.0"

from grecov.bfs import grecov_bfs
from grecov.solver import confidence_interval

__all__ = ["confidence_interval", "grecov_bfs", "__version__"]
