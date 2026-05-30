"""Extremely rare limits.

Source: ...
"""

import datashader as ds
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from loguru import logger

from ..base import SolarDataFrame, SolarSeries
from ..types import QCFlagEnum
from .helpers import (
    construct_qcflag_array,
    MAX_VALUE_COLOR,
    MIN_VALUE_COLOR,
    FAILED_COLOR,
    DENSITY_CMAP,
)

logger.disable(__name__)
logger = logger.opt(colors=True)


def test_ghi(sdf: SolarDataFrame) -> np.ndarray[np.int8]:
    """Evaluate extremely-rare-limits test for GHI."""

    # check that I have what I need
    if "ghi" not in sdf.columns:
        logger.warning("`ghi` column not found in dataframe. Test not possible.")
        return np.full(len(sdf), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)

    # compute whatever I need to apply the test
    ghi = sdf["ghi"]
    min_value = -2.0  # W m-2, to allow for measurement noise when the sun is just below the horizon
    max_value = 50 + 1.20 * sdf.solpos.etn * (sdf.solpos.cosz**1.2)  # W m-2, empirical upper limit

    # compute where the test fails and where it passes
    notna = ghi.notna()
    failed = notna & (ghi.lt(min_value) | ghi.gt(max_value))
    passed = notna & (ghi.ge(min_value) & ghi.le(max_value))

    return construct_qcflag_array(failed, passed)


def plot_test_ghi(sdf: SolarDataFrame, test: SolarSeries, **kwargs) -> plt.Axes:
    """Plot GHI ERL limits and flagged points against solar zenith angle."""

    sdf_ = sdf.assign(
        zenith=sdf.solpos.zenith,
        min_value=-2.0,
        max_value=50 + 1.20 * sdf.solpos.etn * (sdf.solpos.cosz**1.2),
        test=test)

    return plot_test(column="ghi", sdf=sdf_, **kwargs)


def test_dif(sdf: SolarDataFrame) -> np.ndarray[np.int8]:
    """Evaluate extremely-rare-limits test for DIF."""

    # check that I have what I need
    if "dif" not in sdf.columns:
        logger.warning("`dif` column not found in dataframe. Test not possible.")
        return np.full(len(sdf), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)

    # compute whatever I need to apply the test
    dif = sdf["dif"]
    min_value = -2.0  # W m-2
    max_value = 30 + 0.75 * sdf.solpos.etn * (sdf.solpos.cosz**1.2)

    # compute where the test fails and where it passes
    notna = dif.notna()
    failed = notna & (dif.lt(min_value) | dif.gt(max_value))
    passed = notna & (dif.ge(min_value) & dif.le(max_value))

    return construct_qcflag_array(failed, passed)


def plot_test_dif(sdf: SolarDataFrame, test: SolarSeries, **kwargs) -> plt.Axes:
    """Plot DIF ERL limits and flagged points against solar zenith angle."""

    sdf_ = sdf.assign(
        zenith=sdf.solpos.zenith,
        min_value=-2.0,
        max_value=30 + 0.75 * sdf.solpos.etn * (sdf.solpos.cosz**1.2),
        test=test)

    return plot_test(column="dif", sdf=sdf_, **kwargs)


def test_dni(sdf: SolarDataFrame) -> np.ndarray[np.int8]:
    """Evaluate extremely-rare-limits test for DNI."""

    # check that I have what I need
    if "dni" not in sdf.columns:
        logger.warning("`dni` column not found in dataframe. Test not possible.")
        return np.full(len(sdf), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)

    # compute whatever I need to apply the test
    dni = sdf["dni"]
    min_value = -2.0  # W m-2
    max_value = 10 + 0.95 * sdf.solpos.etn * (sdf.solpos.cosz**0.2)

    # compute where the test fails and where it passes
    notna = dni.notna()
    failed = notna & (dni.lt(min_value) | dni.gt(max_value))
    passed = notna & (dni.ge(min_value) & dni.le(max_value))

    return construct_qcflag_array(failed, passed)


def plot_test_dni(sdf: SolarDataFrame, test: SolarSeries, **kwargs) -> plt.Axes:
    """Plot DNI ERL limits and flagged points against solar zenith angle."""

    sdf_ = sdf.assign(
        zenith=sdf.solpos.zenith,
        min_value=-2.0,
        max_value=10 + 0.95 * sdf.solpos.etn * (sdf.solpos.cosz**0.2),
        test=test)

    kwargs.setdefault("rc", {"legend.loc": "lower left"})
    return plot_test(column="dni", sdf=sdf_, **kwargs)


def plot_test(column: str, sdf: SolarDataFrame, **kwargs) -> plt.Axes:
    """Render a standard ERL diagnostic density plot for one irradiance column."""

    plt.style.use("solarpandas-qc")
    if "rc" in kwargs:
        mpl.rcParams.update(kwargs["rc"])

    ax = kwargs.pop("ax", None)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 8), layout="constrained")
    ax_box = ax.get_window_extent()

    title = f"{column.upper()} PPL Test Results"
    network = sdf.custom_metadata.get("network", None)
    if network is not None and network.casefold() == "bsrn":
        station = sdf.custom_metadata.get("station", "unknown station")
        location = sdf.custom_metadata.get("location", "unknown location")
        acronym = sdf.custom_metadata.get("acronym", "unknown acronym")
        title += f" at {station}, {location} ({acronym.upper()}, BSRN)"
    title += f" (lat={sdf.latitude:.4f}, lon={sdf.longitude:.4f}, alt={sdf.elevation:.0f} m)"

    cvs = ds.Canvas(plot_width=int(ax_box.width), plot_height=int(ax_box.height),
                    x_range=(sdf.solpos.zenith.min(), sdf.solpos.zenith.max()),
                    y_range=(-10, sdf[column].max()))

    plt.scatter("zenith", "max_value", data=sdf, label="Max Limit", color=MAX_VALUE_COLOR, s=2)
    plt.scatter("zenith", "min_value", data=sdf, label="Min Limit", color=MIN_VALUE_COLOR, s=2)
    agg = cvs.points(sdf, "zenith", column, ds.count()).pipe(lambda xa: xa.where(xa > 0))
    mesh = ax.pcolormesh(agg.zenith, agg[column], agg.values, cmap=DENSITY_CMAP, norm=plt.cm.colors.LogNorm())
    plt.colorbar(mesh, ax=ax, pad=0.02, label=f"{column.upper()} Counts Density (log scale)")
    plt.scatter("zenith", column, data=sdf.loc[sdf.test.flag.fails], label="Failed Points",
                color=FAILED_COLOR, s=5, zorder=1003)
    plt.xlabel("Solar Zenith Angle (deg)")
    plt.ylabel(f"{column.upper()} (W m$^{{-2}}$)")
    plt.title(title)
    plt.xlim(right=95)
    plt.legend()
    plt.grid()

    return plt.gca()
