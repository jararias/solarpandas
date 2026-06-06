"""Pandas accessors and caches for clear-sky irradiance estimations."""

from functools import lru_cache
from typing import Literal

import numpy as np
import pandas as pd
import spartasolar.atmosphere
from loguru import logger

from ..base import SolarDataFrame, SolarSeries
from ..config import get_option

logger.disable(__name__)
logger = logger.opt(colors=True)


@lru_cache(maxsize=None)
def _compute_cached_clearsky(
    times: bytes, latitude: float, longitude: float, atmosphere: str, model: str
):
    """Compute clear-sky irradiance and cache it by site and timestamps."""
    logger.debug(
        f"evaluating clearsky irradiance with `{model}` model "
        f"and `{atmosphere}` atmosphere..."
    )
    atmos_obj = getattr(spartasolar.atmosphere, atmosphere)
    index = pd.to_datetime(np.frombuffer(times, dtype="datetime64[ns]"))
    args = (pd.DatetimeIndex(index), latitude, longitude)
    try:
        return atmos_obj.at_site(*args).compute(model)
    except AttributeError:
        return atmos_obj.at_sites(*args).compute(model)


def clear_clearsky_cache() -> None:
    """Clear the in-memory clear-sky irradiance cache.

    Examples
    --------
    >>> import solarpandas as sp
    >>> sp.clear_clearsky_cache()
    """
    _compute_cached_clearsky.cache_clear()
    logger.debug("clearsky cache cleared")


def get_clearsky_cache_info():
    """Return cache statistics for clear-sky computations.

    Returns
    -------
    dict[str, int | None]
        Dictionary with ``hits``, ``misses``, ``current_size`` and ``max_size``.

    Examples
    --------
    >>> import solarpandas as sp
    >>> info = sp.get_clearsky_cache_info()
    >>> "current_size" in info
    True
    """
    info = _compute_cached_clearsky.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "current_size": info.currsize,
        "max_size": info.maxsize,
    }


class BaseClearskyIrradianceAccessor:
    """Base class providing cached clear-sky irradiance properties.

    Subclasses configure ``_atmosphere`` and ``_model`` to select the
    atmosphere dataset and irradiance model used for computations.

    Attributes
    ----------
    ghi : SolarSeries
        Clear-sky global horizontal irradiance in W m\u207b\u00b2.
    dni : SolarSeries
        Clear-sky direct normal irradiance in W m\u207b\u00b2.
    dif : SolarSeries
        Clear-sky diffuse horizontal irradiance in W m\u207b\u00b2.
    csi : SolarSeries
        Clear-sky index (ratio of measured to clear-sky GHI).
    """

    def __init__(self, sdf_obj):
        self._sdf = self._validate(sdf_obj)
        self._model = get_option("clearsky.model", default="SPARTA")
        self._atmosphere = get_option("clearsky.atmosphere", default="crs_soda")
        if not hasattr(spartasolar.atmosphere, self._atmosphere):
            raise ValueError(f"invalid clearsky atmosphere `{self._atmosphere}`")
        logger.debug(
            f"initialized {self.__class__.__name__} with `{self._model}` "
            f"model and `{self._atmosphere}` atmosphere"
        )

    @staticmethod
    def _validate(obj):
        if not isinstance(obj, (SolarSeries, SolarDataFrame)):
            name = obj.__class__.__name__
            raise AttributeError(
                f"required a SolarSeries or SolarDataFrame instance. Got {name}"
            )
        return obj

    def _get_cached_clearsky(self, variable: Literal["ghi", "dni", "dif", "csi"]):
        logger.debug(
            f"retrieving `{variable}` from clearsky cache for site at "
            f"({self._sdf.latitude}, {self._sdf.longitude}) and {len(self._sdf.index)} timestamps..."
        )
        logger.debug(f"{self._sdf.index}")
        # To speed up the cache lookup, we convert the times to bytes and use them as part
        # of the cache key (see related notes in the solpos accessor).
        # The numpy datetime64[ns] type does not have timezone information, so we need to
        # convert the times to UTC before converting to bytes. Then, we transform the utc
        # times to tz-naive because the current version of sparta-solar does not support
        # timezone-aware times.
        time_ary_bytes = np.array(
            self._sdf.index.tz_convert("UTC"), dtype="datetime64[ns]"
        ).tobytes()
        clearsky = _compute_cached_clearsky(
            time_ary_bytes,
            self._sdf.latitude,
            self._sdf.longitude,
            self._atmosphere,
            self._model,
        )

        if variable not in clearsky.data_vars:
            raise ValueError(f"clearsky results do not include `{variable}`")

        return SolarSeries(
            clearsky[variable].isel(site=0).to_pandas().values,
            index=self._sdf.index,
            name=variable,
            latitude=self._sdf.latitude,
            longitude=self._sdf.longitude,
            elevation=self._sdf.elevation,
            custom_metadata=self._sdf.custom_metadata,
        )

    @property
    def ghi(self) -> SolarSeries:
        """Clear-sky global horizontal irradiance in W m\u207b\u00b2."""
        return self._get_cached_clearsky("ghi")

    @property
    def dni(self) -> SolarSeries:
        """Clear-sky direct normal irradiance in W m\u207b\u00b2."""
        return self._get_cached_clearsky("dni")

    @property
    def dif(self) -> SolarSeries:
        """Clear-sky diffuse horizontal irradiance in W m\u207b\u00b2."""
        return self._get_cached_clearsky("dif")

    @property
    def csi(self) -> SolarSeries:
        """Clear-sky index (ratio of measured to modelled clear-sky GHI)."""
        return self._get_cached_clearsky("csi")


@pd.api.extensions.register_series_accessor("clearsky")
@pd.api.extensions.register_dataframe_accessor("clearsky")
class ClearskyIrradianceAccessor(BaseClearskyIrradianceAccessor):
    """Accessor for clear-sky irradiance variables (GHI, DNI, DIF, CSI).

    Notes
    -----
    Cached properties use configuration options ``clearsky.model`` and
    ``clearsky.atmosphere``.

    Examples
    --------
    >>> sdf.clearsky.ghi
    >>> sdf.clearsky.compute("crs_soda", "SPARTA").dni
    """

    def compute(
        self,
        atmosphere: Literal[
            "merra2_daily", "merra2_gee", "merra2_lta", "crs_soda", "custom"
        ],
        model: str = "SPARTA",
    ) -> SolarDataFrame:
        """Compute clear-sky irradiance once without using cache.

        Parameters
        ----------
        atmosphere : {"merra2_daily", "merra2_gee", "merra2_lta", "crs_soda", "custom"}
            Atmosphere dataset source.
        model : str, default "SPARTA"
            Irradiance model name accepted by the selected atmosphere backend.

        Returns
        -------
        SolarDataFrame
            Dataframe with columns ``ghi``, ``dni``, ``dif`` and ``csi``
            computed for the requested atmosphere and model."""
        logger.debug(
            f"evaluating clearsky with `{model}` model and `{atmosphere}` atmosphere..."
        )
        if not hasattr(spartasolar.atmosphere, atmosphere):
            raise ValueError(f"invalid clearsky atmosphere `{atmosphere}`")
        if atmosphere == "custom":
            raise NotImplementedError(
                "TODO: implement support for user-provided custom "
                "atmosphere datasets from dataframe columns"
            )
        atmos_obj = getattr(spartasolar.atmosphere, atmosphere)
        args = (
            self._sdf.index.tz_convert("UTC").tz_localize(None),
            self._sdf.latitude,
            self._sdf.longitude,
        )
        try:
            xa_result = atmos_obj.at_site(*args).compute(model)
        except AttributeError:
            xa_result = atmos_obj.at_sites(*args).compute(model)
        df_result = (
            xa_result.isel(site=0).drop_vars(["lat", "lon", "site"]).to_dataframe()
        )
        return self._sdf.replace_data(df_result)


@pd.api.extensions.register_series_accessor("lta")
@pd.api.extensions.register_dataframe_accessor("lta")
class LTAIrradianceAccessor(BaseClearskyIrradianceAccessor):
    """Accessor for long-term-average clear-sky irradiance products.

    Examples
    --------
    >>> sdf.lta.ghi
    >>> sdf.lta.dni
    """

    def __init__(self, sdf_obj):
        self._sdf = self._validate(sdf_obj)
        self._model = get_option("clearsky.model", default="SPARTA")
        self._atmosphere = get_option("clearsky.lta_atmosphere", default="merra2_lta")
        if not hasattr(spartasolar.atmosphere, self._atmosphere):
            raise ValueError(f"invalid clearsky atmosphere `{self._atmosphere}`")


@pd.api.extensions.register_series_accessor("cda")
@pd.api.extensions.register_dataframe_accessor("cda")
class CDAIrradianceAccessor(BaseClearskyIrradianceAccessor):
    """Accessor for clear-day-analysis clear-sky irradiance products.

    Examples
    --------
    >>> sdf.cda.ghi
    >>> sdf.cda.csi
    """

    def __init__(self, sdf_obj):
        self._sdf = self._validate(sdf_obj)
        self._model = get_option("clearsky.model", default="SPARTA")
        self._atmosphere = get_option("clearsky.cda_atmosphere", default="merra2_cda")
        if not hasattr(spartasolar.atmosphere, self._atmosphere):
            raise ValueError(f"invalid clearsky atmosphere `{self._atmosphere}`")
