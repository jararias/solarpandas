

import colorcet as cc
import datashader as ds
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from .helpers import construct_qcflag_array
from ..base import SolarDataFrame, SolarSeries
from ..types import QCFlagEnum

logger.disable(__name__)
logger = logger.opt(colors=True)


def test_Kn_ppl(sdf: SolarDataFrame) -> np.ndarray[np.int8]:
    """Test that Kn is within physically possible limits."""

    # flagKnKt in Table 4 of Forstinger et al.

    # check that I have what I need
    if "ghi" not in sdf.columns:
        logger.warning("`ghi` column not found in dataframe. Test not possible.")
        return np.full(len(sdf), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)

    if "dni" not in sdf.columns:
        logger.warning("`dni` column not found in dataframe. Test not possible.")
        return np.full(len(sdf), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)

    # compute whatever I need to apply the test
    ghi = sdf["ghi"]
    Kn = sdf.param.Kn
    KT = sdf.param.KT
    max_value = KT

    # compute where the test fails and where it passes
    notna = ghi.notna() & Kn.notna() & KT.notna()
    verifiable = notna & ghi.gt(50.) & Kn.gt(0.) & KT.gt(0.)
    failed = verifiable & Kn.ge(max_value)
    passed = verifiable & Kn.lt(max_value)

    return construct_qcflag_array(failed, passed)


def plot_test_Kn_ppl(sdf: SolarDataFrame, test: SolarSeries, **kwargs) -> plt.Axes:

    KT = sdf.param.KT
    Kn = sdf.param.Kn
    max_value = KT
    not_verifiable = sdf["ghi"].le(50.) | Kn.le(0.) | KT.le(0.)

    sdf_ = sdf.assign(
        KT=KT,
        Kn=Kn,
        max_value=max_value,
        not_verifiable=not_verifiable,
        test=test)

    return plot_test(x="KT", y="Kn", sdf=sdf_, **kwargs)


def test_Kn_erl(sdf: SolarDataFrame) -> np.ndarray[np.int8]:
    """Test that Kn is within extremely rare limits."""

    # flagKn in Table 4 of Forstinger et al.

    # check that I have what I need
    if "dni" not in sdf.columns:
        logger.warning("`dni` column not found in dataframe. Test not possible.")
        return np.full(len(sdf), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)

    # compute whatever I need to apply the test
    ghi = sdf["ghi"]
    Kn = sdf.param.Kn
    max_value = (1100. + sdf.elevation*0.03) / sdf.solpos.etn

    # compute where the test fails and where it passes
    notna = ghi.notna() & Kn.notna()
    verifiable = notna & ghi.gt(50.) & Kn.gt(0.)
    failed = verifiable & Kn.ge(max_value)
    passed = verifiable & Kn.lt(max_value)

    return construct_qcflag_array(failed, passed)


def plot_test_Kn_erl(sdf: SolarDataFrame, test: SolarSeries, **kwargs) -> plt.Axes:

    KT = sdf.param.KT
    Kn = sdf.param.Kn
    max_value = (1100. + sdf.elevation*0.03) / sdf.solpos.etn
    not_verifiable = sdf["ghi"].le(50.) | Kn.le(0.)

    sdf_ = sdf.assign(
        KT=KT,
        Kn=Kn,
        max_value=max_value,
        not_verifiable=not_verifiable,
        test=test)

    return plot_test(x="KT", y="Kn", sdf=sdf_, **kwargs)


def test_KT_erl(sdf: SolarDataFrame) -> np.ndarray[np.int8]:
    """Test that KT is within extremely rare limits."""

    # flagKt in Table 4 of Forstinger et al.

    # check that I have what I need
    if "ghi" not in sdf.columns:
        logger.warning("`ghi` column not found in dataframe. Test not possible.")
        return np.full(len(sdf), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)

    # compute whatever I need to apply the test
    ghi = sdf["ghi"]
    KT = sdf.param.KT
    max_value = 1.35

    # compute where the test fails and where it passes
    notna = ghi.notna() & KT.notna()
    verifiable = notna & ghi.gt(50.) & KT.gt(0.)
    failed = verifiable & KT.ge(max_value)
    passed = verifiable & KT.lt(max_value)

    return construct_qcflag_array(failed, passed)


def plot_test_KT_erl(sdf: SolarDataFrame, test: SolarSeries, **kwargs) -> plt.Axes:

    KT = sdf.param.KT
    max_value = KT.clone(1.35)
    not_verifiable = sdf["ghi"].le(50.) | KT.le(0.)

    sdf_ = sdf.assign(
        sza=sdf.solpos.zenith,
        KT=KT,
        max_value=max_value,
        not_verifiable=not_verifiable,
        test=test)

    return plot_test(x="sza", y="KT", sdf=sdf_, rc={"legend.loc": "lower left"})


def test_K_erl(sdf: SolarDataFrame) -> np.ndarray[np.int8]:
    """Test that K is within extremely rare limits."""

    # flagKlowSZA and flagKhighSZA in Table 4 of Forstinger et al.

    # check that I have what I need
    if "ghi" not in sdf.columns:
        logger.warning("`ghi` column not found in dataframe. Test not possible.")
        return np.full(len(sdf), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)

    if "dif" not in sdf.columns:
        logger.warning("`dif` column not found in dataframe. Test not possible.")
        return np.full(len(sdf), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)

    # compute whatever I need to apply the test
    ghi = sdf["ghi"]
    dif = sdf["dif"]
    sza = sdf.solpos.zenith
    K = sdf.param.K
    max_value = ghi.clone(1.05).where(sza.lt(75.), 1.10)

    # compute where the test fails and where it passes
    notna = ghi.notna() & dif.notna() & K.notna()
    verifiable = notna & ghi.gt(50.) & K.gt(0.)
    failed = verifiable & K.ge(max_value)
    passed = verifiable & K.lt(max_value)

    return construct_qcflag_array(failed, passed)


def plot_test_K_erl(sdf: SolarDataFrame, test: SolarSeries, **kwargs) -> plt.Axes:

    K = sdf.param.K
    KT = sdf.param.KT
    max_value = sdf["ghi"].clone(1.05).where(sdf.solpos.sza.lt(75.), 1.10)
    not_verifiable = sdf["ghi"].le(50.) | K.le(0.)

    sdf_ = sdf.assign(
        KT=KT,
        K=K,
        max_value=max_value,
        not_verifiable=not_verifiable,
        test=test)

    return plot_test(x="KT", y="K", sdf=sdf_, rc={"legend.loc": "lower left"})


def test_K_erl_clear(sdf: SolarDataFrame) -> np.ndarray[np.int8]:
    """Test that K is within extremely rare limits."""

    # flagKKt in Table 4 of Forstinger et al.

    # check that I have what I need
    if "ghi" not in sdf.columns:
        logger.warning("`ghi` column not found in dataframe. Test not possible.")
        return np.full(len(sdf), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)

    if "dif" not in sdf.columns:
        logger.warning("`dif` column not found in dataframe. Test not possible.")
        return np.full(len(sdf), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)

    # compute whatever I need to apply the test
    ghi = sdf["ghi"]
    dif = sdf["dif"]
    sza = sdf.solpos.zenith
    K = sdf.param.K
    KT = sdf.param.KT
    max_value = 0.96

    # compute where the test fails and where it passes
    notna = ghi.notna() & dif.notna() & K.notna()
    verifiable = notna & sza.lt(85.) & ghi.gt(150.) & K.gt(0.) & KT.gt(0.6)
    failed = verifiable & K.ge(max_value)
    passed = verifiable & K.lt(max_value)

    return construct_qcflag_array(failed, passed)


def plot_test_K_erl_clear(sdf: SolarDataFrame, test: SolarSeries, **kwargs) -> plt.Axes:

    K = sdf.param.K
    KT = sdf.param.KT
    max_value = sdf["ghi"].clone(0.96).where(sdf.solpos.sza.lt(85.), pd.NA)
    not_verifiable = sdf["ghi"].le(150.) | K.le(0.) | KT.le(0.6)

    sdf_ = sdf.assign(
        K=K,
        KT=KT,
        max_value=max_value,
        not_verifiable=not_verifiable,
        test=test)

    return plot_test(x="KT", y="K", sdf=sdf_, rc={"legend.loc": "lower left"})


def plot_test(x: str, y: str, sdf: SolarDataFrame, **kwargs) -> plt.Axes:

    plt.style.use("solarpandas-qc")
    if "rc" in kwargs:
        mpl.rcParams.update(kwargs["rc"])

    max_value_color = "mistyrose"
    failed_color = "firebrick"
    not_verifiable_color = "navajowhite"
    density_cmap = cc.cm.blues_r

    ax = kwargs.pop("ax", None)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 8), layout="constrained")
    ax_box = ax.get_window_extent()

    title = f"{y} PPL Test Results"
    network = sdf.custom_metadata.get("network", None)
    if network is not None and network.casefold() == "bsrn":
        station = sdf.custom_metadata.get("station", "unknown station")
        location = sdf.custom_metadata.get("location", "unknown location")
        acronym = sdf.custom_metadata.get("acronym", "unknown acronym")
        title += f" at {station}, {location} ({acronym.upper()}, BSRN)"
    title += f" (lat={sdf.latitude:.4f}, lon={sdf.longitude:.4f}, alt={sdf.elevation:.0f} m)"

    _BOUNDS = {
        "KT": (-0.05, 1.4),
        "Kn": (-0.05, 1.15),
        "K": (-0.05, 1.15),
        "sza": (0, 90),
    }

    cvs = ds.Canvas(plot_width=int(ax_box.width), plot_height=int(ax_box.height),
                    x_range=_BOUNDS[x], y_range=_BOUNDS[y])

    plt.scatter(x, "max_value", data=sdf, label="Max. Limit", color=max_value_color, s=1)
    agg = cvs.points(sdf, x, y, ds.count()).pipe(lambda xa: xa.where(xa > 0))
    mesh = ax.pcolormesh(agg[x], agg[y], agg.values, cmap=density_cmap, norm=plt.cm.colors.LogNorm())
    plt.colorbar(mesh, ax=ax, pad=0.02, label=f"{y} Counts Density (log scale)")
    plt.scatter(x, y, data=sdf.loc[sdf.test.flag.fails], label="Failed Points", color=failed_color, s=5)
    plt.scatter(x, y, data=sdf.loc[sdf.not_verifiable], label="Not verifiable", color=not_verifiable_color, s=5)
    plt.xlabel(f"{x} (-)")
    plt.ylabel(f"{y} (-)")
    plt.title(title)
    plt.xlim(_BOUNDS[x])
    plt.ylim(_BOUNDS[y])
    plt.legend()
    plt.grid()

    return plt.gca()

