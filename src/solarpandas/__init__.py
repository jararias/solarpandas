
# ruff: noqa: F401

import importlib.metadata

from . import config, sample_data
from .accessors import (
    SolarPositionAccessor,
    get_solpos_cache_info,
    clear_solpos_cache,
    ClearskyIrradianceAccessor,
    LTAIrradianceAccessor,
    CDAIrradianceAccessor,
    clear_clearsky_cache,
    get_clearsky_cache_info,
    QualityControlAccessor,
    clear_qc_cache,
    get_qc_cache_info,
)
from .base import SolarSeries, SolarDataFrame, read_csv, read_parquet 
from .logtools import enable_logger
from .mplstyles import register as _register_mplstyles

try:
    __version__ = importlib.metadata.version("solarpandas")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

_register_mplstyles()
enable_logger("solarpandas", level="INFO")
