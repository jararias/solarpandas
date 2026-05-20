
# ruff: noqa: F401

import importlib.metadata

from . import config, sample_data
from .accessors import SolarPositionAccessor
from .accessors import (
    ClearskyIrradianceAccessor,
    LTAIrradianceAccessor,
    CDAIrradianceAccessor,
    clear_clearsky_cache,
    get_clearsky_cache_info,
)
from .base import SolarSeries, SolarDataFrame, read_csv, read_parquet 
from .logtools import enable_logger

try:
    __version__ = importlib.metadata.version("solarpandas")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

enable_logger("solarpandas", level="DEBUG")
