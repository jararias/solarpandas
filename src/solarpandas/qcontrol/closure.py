
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
    MIN_VALUE_COLOR,
    FAILED_COLOR,
    NOT_VERIFIABLE_COLOR,
    DENSITY_CMAP,
)

logger.disable(__name__)
logger = logger.opt(colors=True)


def test_closure(sdf: SolarDataFrame) -> np.ndarray[np.int8]:
    """Test that check the closure consistency."""

    # flag3lowSZA and flag3highSZA in Table 5 of Forstinger et al.

    # check that I have what I need
    if "ghi" not in sdf.columns:
        logger.warning("`ghi` column not found in dataframe. Test not possible.")
        return np.full(len(sdf), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)

    if "dni" not in sdf.columns:
        logger.warning("`dni` column not found in dataframe. Test not possible.")
        return np.full(len(sdf), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)

    if "dif" not in sdf.columns:
        logger.warning("`dif` column not found in dataframe. Test not possible.")
        return np.full(len(sdf), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)

    # compute whatever I need to apply the test
    ghi = sdf["ghi"]
    dni = sdf["dni"]
    dif = sdf["dif"]
    ghi_closure = dni * sdf.solpos.cosz + dif
    closure_ratio = (ghi / ghi_closure).where(sdf.solpos.sza.lt(87.), 1.)
    min_value = np.where(sdf.solpos.sza.le(75.), 0.92, 0.85)
    max_value = np.where(sdf.solpos.sza.le(75.), 1.08, 1.15)

    # compute where the test fails and where it passes
    notna = ghi.notna() & dni.notna() & dif.notna()
    verifiable = notna & ghi.gt(50.)
    failed = verifiable & (closure_ratio.lt(min_value) | closure_ratio.gt(max_value))
    passed = verifiable & (closure_ratio.ge(min_value) & closure_ratio.le(max_value))

    return construct_qcflag_array(failed, passed)


def plot_test_closure(sdf: SolarDataFrame, test: SolarSeries, **kwargs) -> plt.Axes:

    plt.style.use("solarpandas-qc")
    mpl.rcParams.update({"legend.loc": "lower left"})

    ax = kwargs.pop("ax", None)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 8), layout="constrained")
    ax_box = ax.get_window_extent()

    title = "Closure Test Results"
    network = sdf.custom_metadata.get("network", None)
    if network is not None and network.casefold() == "bsrn":
        station = sdf.custom_metadata.get("station", "unknown station")
        location = sdf.custom_metadata.get("location", "unknown location")
        acronym = sdf.custom_metadata.get("acronym", "unknown acronym")
        title += f" at {station}, {location} ({acronym.upper()}, BSRN)"
    title += f" (lat={sdf.latitude:.4f}, lon={sdf.longitude:.4f}, alt={sdf.elevation:.0f} m)"

    cvs = ds.Canvas(plot_width=int(ax_box.width), plot_height=int(ax_box.height),
                    x_range=(0., 90.), y_range=(0.80, 1.20))

    ghi = sdf["ghi"]
    dni = sdf["dni"]
    dif = sdf["dif"]
    ghi_closure = dni * sdf.solpos.cosz + dif
    not_verifiable = ghi.le(50.)

    df = sdf.assign(
        zenith=sdf.solpos.sza,
        closure_ratio=(ghi / ghi_closure).where(sdf.solpos.sza.lt(87.), 1.),
        min_value=np.where(sdf.solpos.sza.le(75.), 0.92, 0.85),
        max_value=np.where(sdf.solpos.sza.le(75.), 1.08, 1.15))

    plt.plot("zenith", "max_value", data=df.sort_values("zenith"), label="Max. Limit", color=MAX_VALUE_COLOR, lw=2)
    plt.plot("zenith", "min_value", data=df.sort_values("zenith"), label="Min. Limit", color=MIN_VALUE_COLOR, lw=2)
    agg = cvs.points(df, "zenith", "closure_ratio", ds.count()).pipe(lambda xa: xa.where(xa > 0))
    mesh = ax.pcolormesh(agg["zenith"], agg["closure_ratio"], agg.values, cmap=DENSITY_CMAP, norm=plt.cm.colors.LogNorm())
    plt.colorbar(mesh, ax=ax, pad=0.02, label="Closure Ratio Counts Density (log scale)")
    plt.scatter("zenith", "closure_ratio", data=df.loc[test.flag.fails], label="Failed Points", color=FAILED_COLOR, s=5)
    plt.scatter("zenith", "closure_ratio", data=df.loc[not_verifiable], label="Not verifiable", color=NOT_VERIFIABLE_COLOR, s=5)
    plt.xlabel("Solar Zenith Angle (degrees)")
    plt.ylabel("Closure Ratio (GHI / closure(DNI, DIF))")
    plt.title(title)
    plt.xlim(0., 90.)
    plt.ylim(0.80, 1.20)
    plt.legend()
    plt.grid()

    return plt.gca()

