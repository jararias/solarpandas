"""Pandas accessors and caching helpers for solar-position computations."""

from functools import lru_cache
from typing import Literal

import numpy as np
import pandas as pd
import sunwhere
from loguru import logger

from ..base import SolarDataFrame, SolarSeries
from ..config import get_option

logger.disable(__name__)
logger = logger.opt(colors=True)


@lru_cache(maxsize=None)
def _compute_cached_solpos(
    times: bytes,
    latitude: float,
    longitude: float,
    algorithm: str,
    refraction: bool,
    engine: str,
) -> sunwhere._base.Sunpos:
    """Compute solar position and cache the result for repeated queries."""
    logger.debug(
        f"evaluating solar position with `{algorithm}` algorithm, "
        f"refraction={refraction}, engine=`{engine}`..."
    )
    index = pd.to_datetime(np.frombuffer(times, dtype="datetime64[ns]"), utc=True)
    args = (pd.DatetimeIndex(index), latitude, longitude)
    kwargs = {"algorithm": algorithm, "refraction": refraction, "engine": engine}
    return sunwhere.sites(*args, **kwargs)


class SolarPositionAccessor:
    """Accessor to compute and expose solar-position variables.

    Notes
    -----
    Cached properties use configuration options under ``solar-position``:
    ``algorithm``, ``refraction`` and ``engine``.

    Examples
    --------
    >>> sdf.solpos.zenith  # cached path using config defaults
    >>> sdf.solpos.compute("psa", True, "numexpr").azimuth  # one-off computation
    """

    def __init__(self, sdf_obj):
        self._sdf = self._validate(sdf_obj)
        self._ISC = 1361.1  # W m-2, the solar constant
        self._algorithm = get_option("solar-position.algorithm", default="psa")
        self._refraction = get_option("solar-position.refraction", default=True)
        self._engine = get_option("solar-position.engine", default="numexpr")

    @staticmethod
    def _validate(obj):
        if not isinstance(obj, (SolarSeries, SolarDataFrame)):
            name = obj.__class__.__name__
            raise AttributeError(
                f"required a SolarSeries or SolarDataFrame instance. Got {name}"
            )
        return obj

    @staticmethod
    def clear_cache() -> None:
        """Clear the in-memory solar-position cache.

        Notes
        -----
        Use this when changing model options and forcing a full recomputation.

        Examples
        --------
        >>> import solarpandas as sp
        >>> sp.clear_solpos_cache()
        """
        _compute_cached_solpos.cache_clear()
        logger.debug("solpos cache cleared")

    @staticmethod
    def get_cache_info():
        """Return cache statistics for solar-position computations.

        Returns
        -------
        dict[str, int | None]
            Dictionary with ``hits``, ``misses``, ``current_size`` and ``max_size``.

        Examples
        --------
        >>> import solarpandas as sp
        >>> info = sp.get_solpos_cache_info()
        >>> "hits" in info
        True
        """
        info = _compute_cached_solpos.cache_info()
        return {
            "hits": info.hits,
            "misses": info.misses,
            "current_size": info.currsize,
            "max_size": info.maxsize,
        }

    def compute(
        self, algorithm: str = "psa", refraction: bool = True, engine: str = "numexpr"
    ) -> sunwhere._base.Sunpos:
        """Compute solar position without using the accessor cache.

        Parameters
        ----------
        algorithm : str, default "psa"
            Solar-position algorithm accepted by ``sunwhere``.
        refraction : bool, default True
            Whether to include atmospheric refraction corrections.
        engine : str, default "numexpr"
            Backend engine used by ``sunwhere``.

        Returns
        -------
        sunwhere._base.Sunpos
            ``sunwhere`` result object containing angular and temporal variables.
        """
        logger.debug(
            f"evaluating solar position with `{algorithm}` algorithm, "
            f"refraction={refraction}, engine=`{engine}`..."
        )
        args = (self._sdf.index, self._sdf.latitude, self._sdf.longitude)
        kwargs = {"algorithm": algorithm, "refraction": refraction, "engine": engine}
        return sunwhere.sites(*args, **kwargs)

    def _get_cached_solpos(self, attr_name: str, as_solarseries: bool = True, **kwargs):
        logger.debug(f"accessing cached solar position attribute `{attr_name}`...")

        # N.B. IMPORTANT!!!!
        # To use lru_cache we need to use only hashable arguments to the cached functions.
        # Pandas dataframes and DatetimeIndex objects are not hashable (because they are
        # mutable), and we need to find some way around this.
        # My initial approach was to convert the pandas dataframe DatetimeIndex to a tuple,
        # to make it inmutable, thus hashable. However, this turned to be extremely slow:
        # computing the solar position of one year of minutely data takes ~3 seconds and
        # retrieving the cached calculation took ~2.5 seconds. That is terrible. Most of
        # that time was used to make the DatetimeIndex > tuple > DatetimeIndex conversion.
        # The new approach is to convert the DatetimeIndex to a numpy array of datetime64[ns],
        # and then to bytes, which are directly hashable. This is much faster the making the
        # tuple and still allows retrieving the original times. Now, computing the solar
        # position of one year of minutely data takes ~0.5 seconds, and retrieving a cached
        # result is a matter of only ~0.03 seconds. That is a huge improvement.

        # the numpy datetime64[ns] type does not have timezone information, so we need to
        # convert the times to UTC before converting to bytes
        time_ary_bytes = np.array(
            self._sdf.index.tz_convert("UTC"), dtype="datetime64[ns]"
        ).tobytes()

        solpos = _compute_cached_solpos(
            time_ary_bytes,
            self._sdf.latitude,
            self._sdf.longitude,
            algorithm=self._algorithm,
            refraction=self._refraction,
            engine=self._engine,
        )

        logger.debug(f"retrieved cached solar position. Extracting `{attr_name}`...")
        if callable(data := getattr(solpos, attr_name)):
            data = data(ISC=self._ISC) if attr_name == "eth" else data(**kwargs)

        if "site" in data.dims:
            data = data.isel(site=0)
        data = data.to_pandas().values

        if not as_solarseries:
            return data

        logger.debug(f"constructing SolarSeries for `{attr_name}`...")
        return SolarSeries(
            data,
            index=self._sdf.index,
            latitude=self._sdf.latitude,
            longitude=self._sdf.longitude,
            elevation=self._sdf.elevation,
            custom_metadata=self._sdf.custom_metadata,
        )

    @property
    def sza(self) -> SolarSeries:
        """Solar zenith angle in degrees. Alias for :attr:`zenith`."""
        return self._get_cached_solpos("sza")

    @property
    def zenith(self) -> SolarSeries:
        """Solar zenith angle in degrees at each timestamp.

        Returns
        -------
        SolarSeries
            Values range from 0° (sun overhead) to 180° (sun below horizon).

        Examples
        --------
        >>> sza = sdf.solpos.zenith
        >>> daytime = sza < 90
        """
        return self._get_cached_solpos("zenith")

    @property
    def elevation(self) -> SolarSeries:
        """Solar elevation angle in degrees (complement of zenith: ``90 - zenith``).

        Returns
        -------
        SolarSeries
            Positive values indicate the sun is above the horizon.
        """
        return 90.0 - self.zenith

    @property
    def azimuth(self) -> SolarSeries:
        """Solar azimuth angle in degrees measured clockwise from north.

        Returns
        -------
        SolarSeries
        """
        return self._get_cached_solpos("azimuth")

    @property
    def cosz(self) -> SolarSeries:
        """Cosine of the solar zenith angle.

        Returns
        -------
        SolarSeries
        """
        return self._get_cached_solpos("cosz")

    @property
    def eth(self) -> SolarSeries:
        """Extraterrestrial horizontal irradiance in W m\u207b\u00b2.

        Returns
        -------
        SolarSeries
            Instantaneous irradiance on a horizontal plane at the top of atmosphere.
        """
        return self._get_cached_solpos("eth")

    @property
    def etn(self) -> SolarSeries:
        """Extraterrestrial normal irradiance in W m\u207b\u00b2.

        Returns
        -------
        SolarSeries
            Irradiance on a surface perpendicular to the solar beam (``ISC * ecf``).
        """
        return self._ISC * self.ecf

    @property
    def ecf(self) -> SolarSeries:
        """Earth\u2013Sun distance correction factor (dimensionless).

        Returns
        -------
        SolarSeries
            Ratio of mean to actual Earth\u2013Sun distance squared.
        """
        return self._get_cached_solpos("ecf")

    @property
    def true_solar_time(self) -> SolarSeries:
        """True Solar Time as a tz-aware datetime series.

        Returns
        -------
        SolarSeries
            Timestamps re-expressed in True Solar Time (TST).

        Examples
        --------
        >>> tst_hour = sdf.solpos.true_solar_time.dt.hour
        """
        return self._get_cached_solpos("true_solar_time")

    @property
    def tst(self) -> SolarSeries:
        """Alias for :attr:`true_solar_time`."""
        return self.true_solar_time

    @property
    def true_solar_day(self) -> SolarSeries:
        """True Solar Time floored to the start of each solar day.

        Returns
        -------
        SolarSeries
        """
        return self.true_solar_time.dt.date  # floor("D")

    @property
    def tsd(self) -> SolarSeries:
        """Alias for :attr:`true_solar_day`."""
        return self.true_solar_day

    @property
    def local_solar_time(self) -> SolarSeries:
        """Local Solar Time as a tz-aware datetime series.

        Computed by shifting the index by the longitude-based solar offset
        (``longitude * 4`` minutes).

        Returns
        -------
        SolarSeries
        """
        deltat = pd.Timedelta(self._sdf.longitude * 4, "min")
        return SolarSeries(
            self._sdf.index + deltat,
            index=self._sdf.index,
            latitude=self._sdf.latitude,
            longitude=self._sdf.longitude,
            elevation=self._sdf.elevation,
            custom_metadata=self._sdf.custom_metadata,
        )

    @property
    def lst(self) -> SolarSeries:
        """Alias for :attr:`local_solar_time`."""
        return self.local_solar_time

    def sunrise(self, units: Literal["rad", "deg", "tst", "lst", "utc"] = "utc"):
        """Return sunrise in the selected coordinate system.

        Parameters
        ----------
        units : {"rad", "deg", "tst", "lst", "utc"}, default "utc"
            Units or time reference used by ``sunwhere``.

        Returns
        -------
        SolarSeries
            Sunrise values aligned to the dataframe index.
        """
        sr = self._get_cached_solpos("sunrise", units={"lst": "utc"}.get(units, units))
        if units == "lst":
            # convert from UTC to LST
            deltat = pd.Timedelta(self._sdf.longitude * 4, "min")
            sr = sr + deltat
        return sr

    def sunset(self, units: Literal["rad", "deg", "tst", "lst", "utc"] = "utc"):
        """Return sunset in the selected coordinate system.

        Parameters
        ----------
        units : {"rad", "deg", "tst", "lst", "utc"}, default "utc"
            Units or time reference used by ``sunwhere``.

        Returns
        -------
        SolarSeries
            Sunset values aligned to the dataframe index.
        """
        ss = self._get_cached_solpos("sunset", units={"lst": "utc"}.get(units, units))
        if units == "lst":
            # convert from UTC to LST
            deltat = pd.Timedelta(self._sdf.longitude * 4, "min")
            ss = ss + deltat
        return ss
