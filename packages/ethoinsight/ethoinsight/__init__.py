"""EthoInsight: EthoVision XT behavioral data analysis library.

Provides parsing, metrics computation, statistical testing, chart generation,
and result assessment for EthoVision exported trajectory data.

Usage in DeerFlow sandbox:
    from ethoinsight import parse, metrics, statistics, charts, assess
"""

from ethoinsight import parse, metrics, statistics, charts, assess

__version__ = "0.1.0"

__all__ = ["parse", "metrics", "statistics", "charts", "assess"]
