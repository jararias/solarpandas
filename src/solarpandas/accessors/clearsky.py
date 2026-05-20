
from functools import lru_cache
from typing import Literal

import pandas as pd
import spartasolar.atmosphere
from loguru import logger

from ..base import SolarSeries, SolarDataFrame
from ..config import get_option

logger.disable(__name__)
logger = logger.opt(colors=True)


@lru_cache(maxsize=None)
def _compute_cached_clearsky(
    index: tuple,
    latitude: float,
    longitude: float,
    atmosphere: str,
    model: str):
    """Compute cached clearsky irradiance."""
    logger.debug(f"evaluating clearsky irradiance with `{model}` model and `{atmosphere}` atmosphere...")
    atmos_obj = getattr(spartasolar.atmosphere, atmosphere)
    args = (pd.DatetimeIndex(index), latitude, longitude)
    try:
        return atmos_obj.at_site(*args).compute(model)
    except AttributeError:
        return atmos_obj.at_sites(*args).compute(model)

def clear_clearsky_cache() -> None:
    """Clear the in-memory clearsky irradiance cache.

    Call this to free memory or force recomputation on the next access.

    Example::

        import solarpandas as sp
        sp.clear_clearsky_cache()
    """
    _compute_cached_clearsky.cache_clear()
    logger.debug("clearsky cache cleared")

def get_clearsky_cache_info():
    """Get information about the current state of the clearsky cache.

    Returns:
        dict: A dictionary containing cache statistics such as hits, misses, and current size.

    Example::

        import solarpandas as sp
        info = sp.get_clearsky_cache_info()
        print(info)
    """
    info = _compute_cached_clearsky.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "current_size": info.currsize,
        "max_size": info.maxsize,
    }


class BaseClearskyIrradianceAccessor:
    def __init__(self, sdf_obj):
        self._sdf = self._validate(sdf_obj)
        self._model = get_option("clearsky.model", default="SPARTA")
        self._atmosphere = get_option("clearsky.atmosphere", default="crs_soda")
        if not hasattr(spartasolar.atmosphere, self._atmosphere):
            raise ValueError(f"invalid clearsky atmosphere `{self._atmosphere}`")

    @staticmethod
    def _validate(obj):
        if not isinstance(obj, (SolarSeries, SolarDataFrame)):
            name = obj.__class__.__name__
            raise AttributeError(f"required a SolarSeries or SolarDataFrame instance. Got {name}")
        return obj

    def _get_cached_clearsky(self, variable: Literal["ghi", "dni", "dif", "csi"]):
        clearsky = _compute_cached_clearsky(
            tuple(self._sdf.index),
            self._sdf.latitude,
            self._sdf.longitude,
            self._atmosphere,
            self._model)

        if variable not in clearsky.data_vars:
            raise ValueError(f"clearsky results do not include `{variable}`")

        return SolarSeries(
            clearsky[variable].isel(site=0).to_pandas().values,
            index=self._sdf.index,
            latitude=self._sdf.latitude,
            longitude=self._sdf.longitude,
            elevation=self._sdf.elevation,
            custom_metadata=self._sdf.custom_metadata)

    @property
    def ghi(self):
        return self._get_cached_clearsky("ghi")

    @property
    def dni(self):
        return self._get_cached_clearsky("dni")

    @property
    def dif(self):
        return self._get_cached_clearsky("dif")

    @property
    def csi(self):
        return self._get_cached_clearsky("csi")


@pd.api.extensions.register_series_accessor("clearsky")
@pd.api.extensions.register_dataframe_accessor("clearsky")
class ClearskyIrradianceAccessor(BaseClearskyIrradianceAccessor):
    """General accessor for computing clearsky irradiance.
    
    By default, it caches results using the specified model and atmosphere in config options (`clearsky.model`
    and `clearsky.atmosphere`).

    For a one-off calculations, the ``compute`` method allows bypassing the cache and specifying the model and
    atmosphere directly.
    
    Example:

        # compute with the default model and atmosphere from config options
        data.clearsky.ghi # uses cached results

        # compute with a specific model and atmosphere, bypassing the cache
        data.clearsky.compute("crs_soda", "SPARTA").ghi
    """

    def compute(self, atmosphere: Literal["merra2_daily", "merra2_gee", "merra2_lta", "crs_soda", "custom"], model: str = "SPARTA"):
        logger.debug(f"evaluating clearsky with `{model}` model and `{atmosphere}` atmosphere...")
        if not hasattr(spartasolar.atmosphere, atmosphere):
            raise ValueError(f"invalid clearsky atmosphere `{atmosphere}`")
        if atmosphere == "custom":
            raise NotImplementedError("TODO: implement support for user-provided custom atmosphere datasets from dataframe columns")
        atmos_obj = getattr(spartasolar.atmosphere, atmosphere)
        args = (self._sdf.index, self._sdf.latitude, self._sdf.longitude)
        try:
            return atmos_obj.at_site(*args).compute(model)
        except AttributeError:
            return atmos_obj.at_sites(*args).compute(model)


@pd.api.extensions.register_series_accessor("lta")
@pd.api.extensions.register_dataframe_accessor("lta")
class LTAIrradianceAccessor(BaseClearskyIrradianceAccessor):
    def __init__(self, sdf_obj):
        self._sdf = self._validate(sdf_obj)
        self._model = get_option("clearsky.model", default="SPARTA")
        self._atmosphere = get_option("clearsky.lta_atmosphere", default="merra2_lta")
        if not hasattr(spartasolar.atmosphere, self._atmosphere):
            raise ValueError(f"invalid clearsky atmosphere `{self._atmosphere}`")


@pd.api.extensions.register_series_accessor("cda")
@pd.api.extensions.register_dataframe_accessor("cda")
class CDAIrradianceAccessor(BaseClearskyIrradianceAccessor):
    def __init__(self, sdf_obj):
        self._sdf = self._validate(sdf_obj)
        self._model = get_option("clearsky.model", default="SPARTA")
        self._atmosphere = get_option("clearsky.cda_atmosphere", default="merra2_cda")
        if not hasattr(spartasolar.atmosphere, self._atmosphere):
            raise ValueError(f"invalid clearsky atmosphere `{self._atmosphere}`")
