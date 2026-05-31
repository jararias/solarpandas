"""Physically possible limits (PPL) quality-control checks.

This module implements qcrad physically possible limit tests and plotting
helpers for global, diffuse and direct irradiance components.
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
    """Evaluate physically-possible limits test for GHI."""

    # check that I have what I need
    if "ghi" not in sdf.columns:
        logger.warning("`ghi` column not found in dataframe. Test not possible.")
        return np.full(len(sdf), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)

    # compute whatever I need to apply the test
    ghi = sdf["ghi"]
    etn = sdf.solpos.etn
    cosz = sdf.solpos.cosz
    min_value = -4.0  # W m-2, to allow for measurement noise when the sun is just below the horizon
    max_value = 100 + 1.50 * etn * (cosz**1.2)  # W m-2, empirical upper limit

    # compute where the test fails and where it passes
    notna = ghi.notna()
    failed = notna & (ghi.lt(min_value) | ghi.gt(max_value))
    passed = notna & (ghi.ge(min_value) & ghi.le(max_value))

    return construct_qcflag_array(failed, passed)


def plot_test_ghi(sdf: SolarDataFrame, test: SolarSeries, **kwargs) -> plt.Axes:
    """Plot GHI PPL limits and flagged points against solar zenith angle."""

    sdf_ = sdf.assign(
        zenith=sdf.solpos.zenith,
        min_value=-4.0,
        max_value=100 + 1.50 * sdf.solpos.etn * (sdf.solpos.cosz**1.2),
        test=test)

    return plot_test(column="ghi", sdf=sdf_, **kwargs)


def test_dif(sdf: SolarDataFrame) -> np.ndarray[np.int8]:
    """Evaluate physically-possible limits test for DIF."""

    # check that I have what I need
    if "dif" not in sdf.columns:
        logger.warning("`dif` column not found in dataframe. Test not possible.")
        return np.full(len(sdf), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)

    # compute whatever I need to apply the test
    dif = sdf["dif"]
    min_value = -4.0  # W m-2
    max_value = 50 + 0.95 * sdf.solpos.etn * (sdf.solpos.cosz**1.2)

    # compute where the test fails and where it passes
    notna = dif.notna()
    failed = notna & (dif.lt(min_value) | dif.gt(max_value))
    passed = notna & (dif.ge(min_value) & dif.le(max_value))

    return construct_qcflag_array(failed, passed)


def plot_test_dif(sdf: SolarDataFrame, test: SolarSeries, **kwargs) -> plt.Axes:
    """Plot DIF PPL limits and flagged points against solar zenith angle."""

    sdf_ = sdf.assign(
        zenith=sdf.solpos.zenith,
        min_value=-4.0,
        max_value=50 + 0.95 * sdf.solpos.etn * (sdf.solpos.cosz**1.2),
        test=test)

    return plot_test(column="dif", sdf=sdf_, **kwargs)


def test_dni(sdf: SolarDataFrame) -> np.ndarray[np.int8]:
    """Evaluate physically-possible limits test for DNI."""

    # check that I have what I need
    if "dni" not in sdf.columns:
        logger.warning("`dni` column not found in dataframe. Test not possible.")
        return np.full(len(sdf), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)

    # compute whatever I need to apply the test
    dni = sdf["dni"]
    min_value = -4.0  # W m-2
    max_value = sdf.solpos.etn

    # compute where the test fails and where it passes
    notna = dni.notna()
    failed = notna & (dni.lt(min_value) | dni.gt(max_value))
    passed = notna & (dni.ge(min_value) & dni.le(max_value))

    return construct_qcflag_array(failed, passed)


def plot_test_dni(sdf: SolarDataFrame, test: SolarSeries, **kwargs) -> plt.Axes:
    """Plot DNI PPL limits and flagged points against solar zenith angle."""

    sdf_ = sdf.assign(
        zenith=sdf.solpos.zenith,
        min_value=-4.0,
        max_value=sdf.solpos.etn,
        test=test)

    kwargs.setdefault("rc", {"legend.loc": "lower left"})
    return plot_test(column="dni", sdf=sdf_, **kwargs)


def plot_test(column: str, sdf: SolarDataFrame, **kwargs) -> plt.Axes:
    """Render a standard PPL diagnostic density plot for one irradiance column."""

    plt.style.use("solarpandas-qc")
    if "rc" in kwargs:
        mpl.rcParams.update(kwargs["rc"])

    ax = kwargs.pop("ax", None)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 8), layout="constrained")
    ax_box = ax.get_window_extent()

    title = f"{column.upper()} PPL Test Results"
    if "location" in sdf.custom_metadata:
        title += f" at {sdf.custom_metadata['location']}"
    if "station" in sdf.custom_metadata:
        title += f" ({sdf.custom_metadata['station']}"
        if "network" in sdf.custom_metadata:
            title += f", {sdf.custom_metadata['network']}"
        title += ")"
    else:
        if "network" in sdf.custom_metadata:
            title += f" ({sdf.custom_metadata['network']})"
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



# def test_ghi_polars(sdf: SolarDataFrame):
#     """Test that GHI is within physically-possible limits.
#     Source: ...
#     """
#     import polars as po

#     name = "ghi_ppl"

#     # check that I have what I need: ghi, in this case
#     if "ghi" not in sdf.columns:
#         logger.warning("`ghi` column not found in dataframe. Test not possible.")
#         test_result = np.full(len(sdf), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)
#         return construct_flag_series(sdf, name, test_result)

#     # compute whatever I need to apply the test
#     df = sdf.assign(eth=sdf.solpos.eth, cosz=sdf.solpos.cosz)

#     # convert to polars dataframe
#     df = po.from_pandas(df, include_index=True)

#     # compute where the test can be evaluated (verifiable),
#     # where it fails (failed) and where it passes (passed)
#     max_value_expr = (
#         100 + 1.50 * po.col("eth") * (po.col("cosz")**1.2)
#     ).alias("max_value")

#     verifiable_expr = (
#         po.col("ghi").is_not_null() & po.col("ghi").is_not_nan() &
#         po.col("max_value").is_not_null() & po.col("max_value").is_not_nan()
#     ).alias("verifiable")

#     failed_expr = (
#         verifiable_expr & (po.col("ghi").lt(-4.0) | po.col("ghi").gt(po.col("max_value")))
#     ).alias("failed")

#     passed_expr = (
#         verifiable_expr & (po.col("ghi").ge(-4.0) & po.col("ghi").le(po.col("max_value")))
#     ).alias("passed")

#     result_expr = (
#         po.when(po.col("failed")).then(po.lit(QCFlagEnum.FAILED.value))
#         .when(po.col("passed")).then(po.lit(QCFlagEnum.PASSED.value))
#         .otherwise(po.lit(QCFlagEnum.NOT_VERIFIABLE.value))
#     ).alias(name)

#     test_result = (
#         df.with_columns(max_value_expr)
#         .select(failed_expr, passed_expr)
#         .select(result_expr))

#     test_result = np.asarray(test_result, dtype=np.int8).reshape(-1)
#     return construct_flag_series(sdf, name, test_result)
