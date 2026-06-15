
"""Public exports for quality-control test functions and utilities."""

# ruff: noqa: F401

from .timeshift import check_timeshift

from ..base import SolarDataFrame, SolarSeries
from ..helpers import normalize


def diurnal_availability(sdf: SolarSeries | SolarDataFrame) -> SolarSeries | SolarDataFrame:
    sdf_ = normalize(sdf)
    sza = sdf_.solpos.zenith
    n_available = sdf_.where(sza < 90).groupby(sdf_.solpos.tsd).count()
    max_availability = sza.where(sza < 90).groupby(sdf_.solpos.tsd).count()
    return n_available.divide(max_availability, axis=0)
