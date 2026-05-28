
import datashader as ds
import matplotlib as mpl
import matplotlib.pyplot as plt
from loguru import logger

from ..base import SolarDataFrame

logger.disable(__name__)
logger = logger.opt(colors=True)


def check_timeshift(sdf: SolarDataFrame, column: str = "auto", **kwargs) -> plt.Figure:

    if column == "auto":
        if (column := "dni") not in sdf.columns:
            if (column := "ghi") not in sdf.columns:
                raise ValueError("`dni` and `ghi` not found in dataframe. Test not possible.")

    am = (sdf.solpos.tst.dt.hour < 12) & (sdf.solpos.sza < 90)
    pm = (sdf.solpos.tst.dt.hour >= 12) & (sdf.solpos.sza < 90)

    plt.style.use("solarpandas-qc")
    if "rc" in kwargs:
        mpl.rcParams.update(kwargs.pop("rc"))

    if (ax := kwargs.pop("ax", None)) is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.subplots_adjust(left=0.08, right=0.94, top=0.95, bottom=0.08)

    pos = ax.get_position()  # posicion original
    ax.set_position([pos.x0, pos.y0, pos.width * 0.95, pos.height])  # reducir el ancho del eje
    pos = ax.get_position()  # nueva posicion del eje
    cb_h = pos.height * 0.48  # altura de la barra de color (45% de la altura del eje)
    cb_w = pos.width * 0.025  # ancho de la barra de color (5% del ancho del eje)
    cax_am = plt.axes([pos.x1 + pos.width*0.01, pos.y1 - cb_h, cb_w, cb_h])  # eje para la barra de color
    cax_pm = plt.axes([pos.x1 + pos.width*0.01, pos.y0, cb_w, cb_h])  # eje para la barra de color

    ax_box = ax.get_window_extent()

    title = "Timeshift Check Results"
    network = sdf.custom_metadata.get("network", None)
    if network is not None and network.casefold() == "bsrn":
        station = sdf.custom_metadata.get("station", "unknown station")
        location = sdf.custom_metadata.get("location", "unknown location")
        acronym = sdf.custom_metadata.get("acronym", "unknown acronym")
        title += f" at {station}, {location} ({acronym.upper()}, BSRN)"
    title += f" (lat={sdf.latitude:.4f}, lon={sdf.longitude:.4f}, alt={sdf.elevation:.0f} m)"

    cvs = ds.Canvas(plot_width=int(ax_box.width), plot_height=int(ax_box.height),
                    x_range=(0., 90.)) #, y_range=(0., None))
    agg = (cvs.points(sdf.assign(sza=sdf.solpos.sza).loc[am], "sza", column, ds.count())
           .pipe(lambda xa: xa.where(xa > 0)))
    mesh = ax.pcolormesh(agg.sza, agg[column], agg.values,
                         cmap="Blues_r", norm=mpl.colors.LogNorm())
    plt.colorbar(mesh, cax=cax_am, label="AM Counts Density (log scale)")

    agg = (cvs.points(sdf.assign(sza=sdf.solpos.sza).loc[pm], "sza", column, ds.count())
           .pipe(lambda xa: xa.where(xa > 0)))
    mesh = ax.pcolormesh(agg.sza, agg[column], agg.values,
                         cmap="Oranges_r", norm=mpl.colors.LogNorm())
    plt.colorbar(mesh, cax=cax_pm, label="PM Counts Density (log scale)")

    ax.set_xlabel("Solar Zenith Angle (degrees)")
    ax.set_ylabel(f"{column.upper()} (W m$^{{-2}}$)")
    ax.set_title(title)
    ax.set_xlim(0., 90.)
    ax.set_ylim(0., None)
    ax.grid()

    return plt.gcf()