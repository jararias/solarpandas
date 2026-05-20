
from functools import lru_cache

import pandas as pd
import sunwhere
from loguru import logger

from ..base import SolarSeries, SolarDataFrame
from ..config import get_option

logger.disable(__name__)
logger = logger.opt(colors=True)


@lru_cache(maxsize=None)
def _compute_cached_solpos(
    index: tuple,
    latitude: float,
    longitude: float,
    algorithm: str,
    refraction: bool,
    engine: str
) -> sunwhere._base.Sunpos:
    """Compute cached solar position."""
    logger.debug(f"evaluating solar position with `{algorithm}` algorithm, refraction={refraction}, engine=`{engine}`...")
    args = (pd.DatetimeIndex(index), latitude, longitude)
    kwargs = {"algorithm": algorithm, "refraction": refraction, "engine": engine}
    return sunwhere.sites(*args, **kwargs)

def clear_solpos_cache() -> None:
    """Clear the in-memory solar position cache.

    Call this to free memory or force recomputation on the next access.

    Example::

        import solarpandas as sp
        sp.clear_solpos_cache()
    """
    _compute_cached_solpos.cache_clear()
    logger.debug("solpos cache cleared")

def get_solpos_cache_info():
    """Get information about the current state of the solar position cache.

    Returns:
        dict: A dictionary containing cache statistics such as hits, misses, and current size.

    Example::

        import solarpandas as sp
        info = sp.get_solpos_cache_info()
        print(info)
    """
    info = _compute_cached_solpos.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "current_size": info.currsize,
        "max_size": info.maxsize,
    }


@pd.api.extensions.register_series_accessor("solpos")
@pd.api.extensions.register_dataframe_accessor("solpos")
class SolarPositionAccessor:
    """Accessor for computing solar position.
    
    By default, it caches results using the specified algorithm, refraction, and engine in config options (`solar-position.algorithm`,
    `solar-position.refraction`, and `solar-position.engine`).

    For a one-off calculations, the ``compute`` method allows bypassing the cache and specifying the algorithm, refraction, and engine directly.
    
    Example:

        # compute with the default algorithm, refraction, and engine from config options
        data.solpos.zenith # uses cached results

        # compute with a specific algorithm, refraction, and engine, bypassing the cache
        data.solpos.compute("psa", True, "numexpr").zenith
    """

    def __init__(self, sdf_obj):
        self._sdf = self._validate(sdf_obj)
        self._algorithm = get_option("solar-position.algorithm", default="psa")
        self._refraction = get_option("solar-position.refraction", default=True)
        self._engine = get_option("solar-position.engine", default="numexpr")

    @staticmethod
    def _validate(obj):
        if not isinstance(obj, (SolarSeries, SolarDataFrame)):
            name = obj.__class__.__name__
            raise AttributeError(f"required a SolarSeries or SolarDataFrame instance. Got {name}")
        return obj

    def compute(self, algorithm: str = "psa", refraction: bool = True, engine: str = "numexpr") -> sunwhere._base.Sunpos:
        logger.debug(f"evaluating solar position with `{algorithm}` algorithm, refraction={refraction}, engine=`{engine}`...")
        args = (self._sdf.index, self._sdf.latitude, self._sdf.longitude)
        kwargs = {"algorithm": algorithm, "refraction": refraction, "engine": engine}
        return sunwhere.sites(*args, **kwargs)

    def _get_cached_solpos(self, attr_name: str, as_solarseries: bool = True):
        solpos = _compute_cached_solpos(
            tuple(self._sdf.index),
            self._sdf.latitude,
            self._sdf.longitude,
            algorithm=self._algorithm,
            refraction=self._refraction,
            engine=self._engine)

        if callable(data := getattr(solpos, attr_name)):
            data = data()

        if "site" in data.dims:
            data = data.isel(site=0)
        data = data.to_pandas().values
    
        if not as_solarseries:
            return data

        return SolarSeries(
            data,
            index=self._sdf.index,
            latitude=self._sdf.latitude,
            longitude=self._sdf.longitude,
            elevation=self._sdf.elevation,
            custom_metadata=self._sdf.custom_metadata)

    @property
    def sza(self):
        return self._get_cached_solpos("sza")

    @property
    def zenith(self):
        return self._get_cached_solpos("zenith")

    @property
    def azimuth(self):
        return self._get_cached_solpos("azimuth")

    @property
    def cosz(self):
        return self._get_cached_solpos("cosz")

    @property
    def eth(self):
        return self._get_cached_solpos("eth")

    @property
    def ecf(self):
        return self._get_cached_solpos("ecf")

    @property
    def true_solar_time(self):
        return self._get_cached_solpos("true_solar_time")

    @property
    def tst(self):
        return self.true_solar_time

    @property
    def true_solar_day(self):
        return self.true_solar_time.dt.floor("D")

    @property
    def tsd(self):
        return self.true_solar_day

    @property
    def local_solar_time(self):
        deltat = pd.Timedelta(self._sdf.longitude * 4, "min")
        return SolarSeries(
            self._sdf.index + deltat,
            index=self._sdf.index,
            latitude=self._sdf.latitude,
            longitude=self._sdf.longitude,
            elevation=self._sdf.elevation,
            custom_metadata=self._sdf.custom_metadata)

    @property
    def lst(self):
        return self.local_solar_time
