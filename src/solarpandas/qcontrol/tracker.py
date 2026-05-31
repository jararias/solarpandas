
"""Quality-control checks related to tracker behavior and geometry."""

# Implementations:
# https://git.sophia.minesparis.psl.eu/yves-marie.saint-drenan/libinsitu/-/blob/main/libinsitu/qc_utils.py?ref_type=heads
# https://github.com/dazhiyang/bsrn/blob/main/src/bsrn/qc/tracker.py


import datashader as ds
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from ..base import SolarDataFrame, SolarSeries
from ..types import QCFlagEnum
from .helpers import (
    construct_qcflag_array,
    FAILED_COLOR,
    NOT_VERIFIABLE_COLOR,
    DENSITY_CMAP,
)

logger.disable(__name__)
logger = logger.opt(colors=True)


def test_trackeroff(sdf: SolarDataFrame) -> np.ndarray[np.int8]:
    """Evaluate tracker-off condition test based on CDA ratios."""

    # flagTracker in Table 8 of Forstinger et al.

    # check that I have what I need
    if "ghi" not in sdf.columns:
        logger.warning("`ghi` column not found in dataframe. Test not possible.")
        return np.full(len(sdf), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)

    if "dni" not in sdf.columns:
        logger.warning("`dni` column not found in dataframe. Test not possible.")
        return np.full(len(sdf), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)

    # compute whatever I need to apply the test
    ghi = sdf["ghi"]
    ghic = sdf.cda.ghi
    ghi_ratio = (ghic - ghi) / (ghic + ghi)
    max_value = np.full(len(sdf),0.20)

    dni = sdf["dni"]
    dnic = sdf.cda.dni
    dni_ratio = (dnic - dni) / (dnic + dni)
    min_value = np.full(len(sdf), 0.95)

    # compute where the test fails and where it passes
    notna = ghi.notna() & dni.notna()
    verifiable = notna & sdf.solpos.sza.lt(85.)
    failed = verifiable & dni_ratio.gt(min_value) & ghi_ratio.lt(max_value)
    passed = verifiable & (dni_ratio.le(min_value) | ghi_ratio.ge(max_value))

    return construct_qcflag_array(failed, passed)


def plot_test_trackeroff(sdf: SolarDataFrame, test: SolarSeries, **kwargs) -> plt.Axes:
    """Plot tracker-off diagnostics in KT-K space with flagged points."""

    plt.style.use("solarpandas-qc")
    mpl.rcParams.update({"legend.loc": "lower left"})

    ax = kwargs.pop("ax", None)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 8), layout="constrained")
    ax_box = ax.get_window_extent()

    title = "Tracker-off Test Results"
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
                    x_range=(-0.05, 1.40), y_range=(-0.05, 1.15))

    not_verifiable = sdf.solpos.sza.ge(85.)

    df = sdf.assign(
        KT=sdf.param.KT,
        K=sdf.param.K)

    agg = cvs.points(df, "KT", "K", ds.count()).pipe(lambda xa: xa.where(xa > 0))
    mesh = ax.pcolormesh(agg["KT"], agg["K"], agg.values, cmap=DENSITY_CMAP,
                         norm=plt.cm.colors.LogNorm(), zorder=1000)
    plt.colorbar(mesh, ax=ax, pad=0.02, label="K Counts Density (log scale)")
    plt.scatter("KT", "K", data=df.loc[test.flag.fails], label="Failed Points",
                color=FAILED_COLOR, s=3, zorder=1002)
    plt.scatter("KT", "K", data=df.loc[not_verifiable], label="Not verifiable",
                color=NOT_VERIFIABLE_COLOR, s=3, zorder=1001)
    plt.xlabel("KT (-)")
    plt.ylabel("K (-)")
    plt.title(title)
    plt.xlim(-0.05, 1.40)
    plt.ylim(-0.05, 1.15)
    leg =plt.legend()
    leg.set_zorder(1003)
    plt.grid()

    return plt.gca()
