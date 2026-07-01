
"""Top-level package exports and version metadata for solarpandas."""

# ruff: noqa: F401

import importlib.metadata

from . import config, sample_data

from .accessors import (
    ClearskyIrradianceAccessor,
    CDAIrradianceAccessor,
    LTAIrradianceAccessor,
    ParameterAccessor,
    QualityControlAccessor,
    SkyClassAccessor,
    SolarPlotAccessor,
    SolarPositionAccessor,
)

from .accessors.clearsky import (
    clear_cache as clear_clearsky_cache,
    get_cache_info as get_clearsky_cache_info,
)

from .accessors.qcontrol import (
    clear_cache as clear_qc_cache,
    get_cache_info as get_qc_cache_info,
)

from .accessors.solpos import (
    clear_cache as clear_solpos_cache,
    get_cache_info as get_solpos_cache_info,
)

from .base import SolarSeries, SolarDataFrame
from .iohelpers import read_csv, read_parquet
from .logtools import enable_logger
from .mplstyles import register_mplstyles as _register_mplstyles

try:
    __version__ = importlib.metadata.version("solarpandas")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

_register_mplstyles()  # OPTIMIZE: this line adds a latency of about 0.7 seconds at import time
enable_logger("solarpandas", level="INFO")
