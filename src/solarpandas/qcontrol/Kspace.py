
"""Quality-control tests based on K-space consistency relationships."""

import datashader as ds
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from ..base import SolarDataFrame, SolarSeries
from ..types import QCFlagEnum
from .helpers import (
    construct_qcflag_array,
    MAX_VALUE_COLOR,
    FAILED_COLOR,
    NOT_VERIFIABLE_COLOR,
    DENSITY_CMAP,
)

logger.disable(__name__)
logger = logger.opt(colors=True)


def test_Kn_ppl(sdf: SolarDataFrame) -> np.ndarray[np.int8]:
    """Evaluate the physically possible limits test for Kn."""

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
    """Plot Kn versus KT for physically-possible-limit diagnostics."""

    KT = sdf.param.KT
    Kn = sdf.param.Kn

    sdf_ = sdf.assign(
        KT=KT,
        Kn=Kn,
        max_value=KT,
        not_verifiable=sdf["ghi"].le(50.) | Kn.le(0.) | KT.le(0.),
        test=test)

    kwargs.setdefault("rc", {"legend.loc": "upper left"})
    return plot_test(x="KT", y="Kn", sdf=sdf_, **kwargs)


def test_Kn_erl(sdf: SolarDataFrame) -> np.ndarray[np.int8]:
    """Evaluate the extremely rare limits test for Kn."""

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
    """Plot Kn versus KT for extremely-rare-limit diagnostics."""

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

    kwargs.setdefault("max_value_artist", "scatter")
    return plot_test(x="KT", y="Kn", sdf=sdf_, **kwargs)


def test_KT_erl(sdf: SolarDataFrame) -> np.ndarray[np.int8]:
    """Evaluate the extremely rare limits test for KT."""

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
    """Plot KT against zenith with ERL threshold and flagged points."""

    KT = sdf.param.KT

    sdf_ = sdf.assign(
        zenith=sdf.solpos.zenith,
        KT=KT,
        max_value=np.full(len(KT), 1.35),
        not_verifiable=sdf["ghi"].le(50.) | KT.le(0.),
        test=test)

    return plot_test(x="zenith", y="KT", sdf=sdf_, rc={"legend.loc": "lower left"})


def test_K_erl(sdf: SolarDataFrame) -> np.ndarray[np.int8]:
    """Evaluate the extremely rare limits test for K."""

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
    max_value = ghi.replace_data(1.05).where(sza.lt(75.), 1.10)

    # compute where the test fails and where it passes
    notna = ghi.notna() & dif.notna() & K.notna()
    verifiable = notna & ghi.gt(50.) & K.gt(0.)
    failed = verifiable & K.ge(max_value)
    passed = verifiable & K.lt(max_value)

    return construct_qcflag_array(failed, passed)


def plot_test_K_erl(sdf: SolarDataFrame, test: SolarSeries, **kwargs) -> plt.Axes:
    """Plot K against zenith with ERL threshold and flagged points."""

    K = sdf.param.K

    sdf_ = sdf.assign(
        zenith=sdf.solpos.zenith,
        K=K,
        max_value=np.where(sdf.solpos.zenith.lt(75.), 1.05, 1.10),
        not_verifiable=sdf["ghi"].le(50.) | K.le(0.),
        test=test)

    return plot_test(x="zenith", y="K", sdf=sdf_, rc={"legend.loc": "center left"})


def test_K_erl_clear(sdf: SolarDataFrame) -> np.ndarray[np.int8]:
    """Evaluate K clear-sky consistency test under ERL conditions."""

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
    """Plot K versus KT for ERL clear-sky consistency diagnostics."""

    K = sdf.param.K
    KT = sdf.param.KT

    sdf_ = sdf.assign(
        K=K,
        KT=KT,
        max_value=np.where(sdf.solpos.zenith.lt(85.) & KT.gt(0.6), 0.96, float("nan")),
        not_verifiable=sdf["ghi"].le(150.) | K.le(0.),
        test=test)

    return plot_test(x="KT", y="K", sdf=sdf_, rc={"legend.loc": "lower left"})


def plot_test(x: str, y: str, sdf: SolarDataFrame, **kwargs) -> plt.Axes:
    """Render a generic K-space QC diagnostic plot with density and flags."""

    plt.style.use("solarpandas-qc")
    if "rc" in kwargs:
        mpl.rcParams.update(kwargs["rc"])

    ax = kwargs.pop("ax", None)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 8), layout="constrained")
    ax_box = ax.get_window_extent()

    title = f"{y} PPL Test Results"
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

    _BOUNDS = {
        "KT": (-0.05, 1.4),
        "Kn": (-0.05, 1.15),
        "K": (-0.05, 1.15),
        "zenith": (0, 90),
    }

    cvs = ds.Canvas(plot_width=int(ax_box.width), plot_height=int(ax_box.height),
                    x_range=_BOUNDS[x], y_range=_BOUNDS[y])

    if "max_value_artist" in kwargs and kwargs["max_value_artist"] == "scatter":
        plt.scatter(x, "max_value", data=sdf.sort_values(x), label="Max. Limit",
                    color=MAX_VALUE_COLOR, s=2, zorder=1000)
    else:
        plt.plot(x, "max_value", data=sdf.sort_values(x), label="Max. Limit",
                 color=MAX_VALUE_COLOR, lw=2, zorder=1000)
    agg = cvs.points(sdf, x, y, ds.count()).pipe(lambda xa: xa.where(xa > 0))
    mesh = ax.pcolormesh(agg[x], agg[y], agg.values, cmap=DENSITY_CMAP,
                         norm=plt.cm.colors.LogNorm(), zorder=1001)
    plt.colorbar(mesh, ax=ax, pad=0.02, label=f"{y} Counts Density (log scale)")
    plt.scatter(x, y, data=sdf.loc[sdf.test.flag.fails], label="Failed Points",
                color=FAILED_COLOR, s=5, zorder=1003)
    plt.scatter(x, y, data=sdf.loc[sdf.not_verifiable], label="Not verifiable",
                color=NOT_VERIFIABLE_COLOR, s=3, zorder=1002)
    plt.xlabel(f"{x} (-)")
    plt.ylabel(f"{y} (-)")
    plt.title(title)
    plt.xlim(_BOUNDS[x])
    plt.ylim(_BOUNDS[y])
    leg = plt.legend()
    leg.set_zorder(1004)
    plt.grid()

    return plt.gca()

