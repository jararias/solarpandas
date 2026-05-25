"""Physically-possible limits.

Source: ...
"""

import colorcet as cc
import datashader as ds
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from loguru import logger

from ..base import SolarDataFrame, SolarSeries
from .dtype import QCFlagEnum

logger.disable(__name__)
logger = logger.opt(colors=True)


def test_ghi(sdf: SolarDataFrame) -> np.ndarray[np.int8]:
    """Test that GHI is within physically-possible limits."""
    # check that I have what I need: ghi, in this case
    if "ghi" not in sdf.columns:
        logger.warning("`ghi` column not found in dataframe. Test not possible.")
        return np.full(len(sdf), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)

    # compute whatever I need to apply the test
    ghi = sdf["ghi"]
    eth = sdf.solpos.eth
    cosz = sdf.solpos.cosz
    min_value = -4.0  # W m-2, to allow for measurement noise when the sun is just below the horizon
    max_value = 100 + 1.50 * eth * (cosz**0.2)  # W m-2, empirical upper limit

    # compute where the test can be evaluated (verifiable),
    # where it fails (failed) and where it passes (passed)
    verifiable = ghi.notna() & max_value.notna()
    failed = verifiable & (ghi.lt(min_value) | ghi.gt(max_value))
    passed = verifiable & (ghi.ge(min_value) & ghi.le(max_value))

    # construct the flag: -1 for failed, 0 for not verifiable, 1 for passed
    test_result = np.full(len(ghi), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)
    test_result[verifiable & failed] = QCFlagEnum.FAILED.value
    test_result[verifiable & passed] = QCFlagEnum.PASSED.value

    return test_result


def plot_test_ghi(sdf: SolarDataFrame, flag: SolarSeries, **kwargs) -> plt.Axes:
    """Plot the GHI test limits and results for visual inspection."""

    eth = sdf.solpos.eth
    cosz = sdf.solpos.cosz
    min_value = -4.0  # W m-2, to allow for measurement noise when the sun is just below the horizon
    max_value = 100 + 1.50 * eth * (cosz**0.2)  # W m-2, empirical upper limit

    mpl.rcParams["axes.titlesize"] = 14
    mpl.rcParams["axes.labelsize"] = 12
    mpl.rcParams["xtick.labelsize"] = 12
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["legend.fontsize"] = 11

    ax = kwargs.pop("ax", None)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), layout="constrained")
    ax_box = ax.get_window_extent()

    df = (sdf.assign(zenith=sdf.solpos.zenith, min_value=min_value, max_value=max_value, flag=flag)
          .pipe(lambda df: df.loc[df.zenith < 89]))

    cvs = ds.Canvas(plot_width=int(ax_box.width), plot_height=int(ax_box.height),
                    x_range=(df.zenith.min(), df.zenith.max()), y_range=(-10, df.ghi.max()))

    colors = iter(cc.glasbey_hv)
    agg = cvs.points(df, "zenith", "ghi", ds.count()).pipe(lambda xa: xa.where(xa > 0))
    mesh = ax.pcolormesh(agg.zenith, agg.ghi, agg.values, cmap="viridis", norm=plt.cm.colors.LogNorm())
    plt.colorbar(mesh, ax=ax, label="GHI Density (log scale)")
    plt.scatter("zenith", "max_value", data=df, label="Max Limit", color=next(colors), s=1)
    plt.scatter("zenith", "min_value", data=df, label="Min Limit", color=next(colors), s=1)
    plt.scatter("zenith", "ghi", data=df.loc[flag.fails], label="Failed Points", color="red", s=5)
    plt.xlabel("Solar Zenith Angle (deg)")
    plt.ylabel("GHI (W m-2)")
    plt.title("GHI PPL Test Results")
    plt.xlim(right=95)
    plt.legend()
    plt.grid()

    return plt.gca()


def test_dif(sdf: SolarDataFrame) -> np.ndarray[np.int8]:
    """Test that DIF is within physically-possible limits.
    """

    # check that I have what I need: dif, in this case
    if "dif" not in sdf.columns:
        logger.warning("`dif` column not found in dataframe. Test not possible.")
        return np.full(len(sdf), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)

    # compute whatever I need to apply the test
    dif = sdf["dif"]
    eth = sdf.solpos.eth
    cosz = sdf.solpos.cosz
    min_value = -4.0  # W m-2
    max_value = 50 + 0.95 * eth * (cosz**0.2)

    # compute where the test can be evaluated (verifiable),
    # where it fails (failed) and where it passes (passed)
    verifiable = dif.notna() & max_value.notna()
    failed = verifiable & (dif.lt(min_value) | dif.gt(max_value))
    passed = verifiable & (dif.ge(min_value) & dif.le(max_value))

    # construct the flag: -1 for failed, 0 for not verifiable, 1 for passed
    test_result = np.full(len(dif), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)
    test_result[verifiable & failed] = QCFlagEnum.FAILED.value
    test_result[verifiable & passed] = QCFlagEnum.PASSED.value

    return test_result


def plot_test_dif(sdf: SolarDataFrame, flag: SolarSeries) -> plt.Axes:
    """Plot the DIF test limits and results for visual inspection."""

    dif = sdf["dif"]
    eth = sdf.solpos.eth
    cosz = sdf.solpos.cosz
    min_value = -4.0  # W m-2
    max_value = 50 + 0.95 * eth * (cosz**0.2)

    zenith = sdf.solpos.zenith
    plt.figure(figsize=(10, 8), layout="constrained")
    plt.scatter(zenith, dif, label="DIF", color="C0", s=1)
    plt.scatter(zenith, np.full(len(dif), min_value), label="Min Limit", color="C1", s=1)
    plt.scatter(zenith, max_value, label="Max Limit", color="C2", s=1)
    plt.scatter(zenith[flag.fails], dif[flag.fails], label="Failed Points", color="red", s=5)
    plt.scatter(zenith[flag.not_verifiable], dif[flag.not_verifiable], label="Not Verifiable Points", color="gray", s=5)
    plt.xlabel("Solar Zenith Angle (deg)")
    plt.ylabel("DIF (W m-2)")
    plt.title("DIF PPL Test Results")
    plt.xlim(right=95)
    plt.legend()
    plt.grid()

    return plt.gca()


def test_dni(sdf: SolarDataFrame) -> np.ndarray[np.int8]:
    """Test that DNI is within physically-possible limits.
    """

    # check that I have what I need: dni, in this case
    if "dni" not in sdf.columns:
        logger.warning("`dni` column not found in dataframe. Test not possible.")
        return np.full(len(sdf), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)

    # compute whatever I need to apply the test
    dni = sdf["dni"]
    eth = sdf.solpos.eth
    min_value = -4.0  # W m-2
    max_value = (eth / sdf.solpos.cosz).where(sdf.solpos.cosz > 1e-6, 0.0)

    # compute where the test can be evaluated (verifiable),
    # where it fails (failed) and where it passes (passed)
    verifiable = dni.notna()
    failed = verifiable & (dni.lt(min_value) | dni.gt(max_value))
    passed = verifiable & (dni.ge(min_value) & dni.le(max_value))

    # construct the flag: -1 for failed, 0 for not verifiable, 1 for passed
    test_result = np.full(len(dni), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)
    test_result[verifiable & failed] = QCFlagEnum.FAILED.value
    test_result[verifiable & passed] = QCFlagEnum.PASSED.value

    return test_result


def plot_test_dni(sdf: SolarDataFrame, flag: SolarSeries) -> plt.Axes:
    """Plot the DNI test limits and results for visual inspection."""

    dni = sdf["dni"]
    eth = sdf.solpos.eth
    min_value = -4.0  # W m-2
    max_value = (eth / sdf.solpos.cosz).where(sdf.solpos.cosz > 1e-3, 0.0)

    zenith = sdf.solpos.zenith
    plt.figure(figsize=(10, 8), layout="constrained")
    plt.scatter(zenith, dni, label="DNI", color="C0", s=1)
    plt.scatter(zenith, np.full(len(dni), min_value), label="Min Limit", color="C1", s=1)
    plt.scatter(zenith, max_value, label="Max Limit", color="C2", s=1)
    plt.scatter(zenith[flag.fails], dni[flag.fails], label="Failed Points", color="red", s=5)
    plt.scatter(zenith[flag.not_verifiable], dni[flag.not_verifiable], label="Not Verifiable Points", color="gray", s=5)
    plt.xlabel("Solar Zenith Angle (deg)")
    plt.ylabel("DNI (W m-2)")
    plt.title("DNI PPL Test Results")
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
