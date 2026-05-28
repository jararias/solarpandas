
from typing import Callable, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.dates import DateFormatter

from ..base import SolarDataFrame, SolarSeries
from ..helpers import infer_time_step

logger.disable(__name__)
logger = logger.opt(colors=True)


@pd.api.extensions.register_series_accessor("solarplot")
@pd.api.extensions.register_dataframe_accessor("solarplot")
class SolarPlotAccessor:
    def __init__(self, sdf_obj):
        self._sdf = self._validate(sdf_obj)

    @staticmethod
    def _validate(obj):
        if not isinstance(obj, (SolarSeries, SolarDataFrame)):
            name = obj.__class__.__name__
            raise AttributeError(f"required a SolarSeries orSolarDataFrame instance. Got {name}")
        return obj

    def dtmap(
        self,
        column: str | None = None,
        time_ref: Literal["lst", "tst", "lat", "utc"] = "tst",
        max_sza: float | None = 90.0,
        colorbar: bool = True,
        colorbar_title: str | None = None,
        twilight_line: bool = False,
        twilight_line_kwargs: dict | None = None,
        aggfunc: str | Callable = "mean",
        **kwargs
    ) -> plt.Figure:

        YLABEL_MAPPING = {
            "lst": "Local Solar Time",
            "tst": "True Solar Time",
            "lat": "Local Apparent Time",
            "utc": "Coordinated Universal Time",
        }

        def time_to_minutes(time_obj):
            return int(time_obj.hour * 60 + time_obj.minute + time_obj.second / 60)

        if isinstance(self._sdf, SolarSeries):
            if column is not None:
                logger.warning("Column name ignored when plotting a SolarSeries.")
            column = self._sdf.name or "_unnamed_"
            sdf = self._sdf.to_frame(column)
        else:
            if column is None:
                logger.warning("No column specified for plotting. Defaulting to the first column.")
                column = self._sdf.columns[0]
            elif column not in self._sdf.columns:
                logger.warning(f"Column '{column}' not found in dataframe. Defaulting to the first column.")
                column = self._sdf.columns[0]
            sdf = self._sdf[[column]]

        df = pd.DataFrame(sdf.where(self._sdf.solpos.zenith < (max_sza or 180.), pd.NA))

        time_step = infer_time_step(df)

        if time_ref.casefold() == "lst":
            df = df.set_index(self._sdf.solpos.lst)
        elif time_ref.casefold() in ("tst", "lat"):
            df = df.set_index(self._sdf.solpos.tst)
        else:
            df = df.set_index(self._sdf.index.tz_convert("UTC").tz_localize(None))

        if time_ref.casefold() in ("lst", "tst", "lat"):
            df = df.set_index(df.index.round(time_step))

        table = (
            df.assign(date=df.index.date, time=df.index.time)
            .pivot_table(index="time", columns="date", values=column, aggfunc=aggfunc))

        date_coords = table.columns
        time_coords = table.index.map(lambda t: np.datetime64(time_to_minutes(t), "m"))

        plt.style.use("solarpandas-dtmap")
        if "rc" in kwargs:
            mpl.rcParams.update(kwargs.pop("rc"))

        if (ax := kwargs.pop("ax", None)) is None:
            _, ax = plt.subplots(1, 1, figsize=(12, 6), layout="constrained")

        mesh = ax.pcolormesh(date_coords, time_coords, table.values, **kwargs)
        if colorbar:
            plt.colorbar(mesh, ax=ax, label=colorbar_title or column, pad=0.01,
                         fraction=0.025, shrink=1.0)

        if twilight_line:

            def get_twilight(which: str):
                twilight = (getattr(self._sdf.solpos, which)(units=time_ref)
                            .resample("D").median()
                            .dt.round("1s")
                            .dt.time.map(lambda t: np.datetime64(time_to_minutes(t), "m")))
                twilight = twilight.set_axis(twilight.index.date)
                return twilight.reindex(date_coords, method="nearest", tolerance=pd.Timedelta("1D"))

            default_twilight_kwargs = {"color": "purple", "ls": "--", "lw": 1.5}
            twilight_line_kwargs = default_twilight_kwargs | (twilight_line_kwargs or {})
            ax.plot(date_coords, get_twilight("sunrise"), label="Sunrise", **twilight_line_kwargs)
            ax.plot(date_coords, get_twilight("sunset"), label="Sunset", **twilight_line_kwargs)

        ax.set_xlabel("Date")

        ax.yaxis.set_major_formatter(DateFormatter("%H:%M"))
        ax.set_ylim(np.datetime64(0, "m"), np.datetime64(24 * 60, "m"))
        ax.set_ylabel(YLABEL_MAPPING.get(time_ref.casefold()))

        return ax.get_figure()
