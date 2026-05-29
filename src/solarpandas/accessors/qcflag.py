import inspect

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from matplotlib.colors import BoundaryNorm, ListedColormap

from ..base import SolarDataFrame, SolarSeries
from ..mplstyles import QC_COLOR_FAILED, QC_COLOR_PASSED, QC_COLOR_NOT_VERIFIABLE
from ..qcontrol import helpers, qcrad
from ..types import QCFlagDtype, QCFlagEnum

logger.disable(__name__)
logger = logger.opt(colors=True)


@pd.api.extensions.register_series_accessor("flag")
class QCFlagAccessor:
    """Accessor for Series with QCFlagDtype dtype.

    Usage
    -----
    s.flag.fails           # boolean Series: True where value is -1
    s.flag.passes          # boolean Series: True where value is 1
    s.flag.not_verifiable  # boolean Series: True where value is 0
    """

    def __init__(self, series: pd.Series | SolarSeries) -> None:
        if not isinstance(series.dtype, QCFlagDtype):
            raise AttributeError(
                "The .flag accessor is only available for Series with dtype 'QCFlagDtype'."
            )
        self._series = series

    @property
    def fails(self) -> pd.Series | SolarSeries:
        flag = self._series.array.fails.astype(bool)
        if isinstance(self._series, SolarSeries):
            return self._series.clone(other=flag).rename(self._series.name)
        return pd.Series(flag, index=self._series.index, name=self._series.name)

    @property
    def passes(self) -> pd.Series | SolarSeries:
        flag = self._series.array.passes.astype(bool)
        if isinstance(self._series, SolarSeries):
            return self._series.clone(other=flag).rename(self._series.name)
        return pd.Series(flag, index=self._series.index, name=self._series.name)

    @property
    def not_verifiable(self) -> pd.Series | SolarSeries:
        flag = self._series.array.not_verifiable.astype(bool)
        if isinstance(self._series, SolarSeries):
            return self._series.clone(other=flag).rename(self._series.name)
        return pd.Series(flag, index=self._series.index, name=self._series.name)

    def counts(self, skip_nighttime: bool = True, **kwargs) -> pd.Series:
        """Count occurrences of each flag value (-1, 0, 1, NA)."""
        series = self._series
        if skip_nighttime:
            if not isinstance(series, SolarSeries):
                logger.warning(
                    "skip_nighttime=True is only valid for SolarSeries. Skipping nighttime filtering."
                )
            else:
                series = series.loc[series.solpos.zenith < 90]
        return series.value_counts(**kwargs).rename(
            index={e.value: e.name for e in QCFlagEnum}
        )

    def pieplot(self, skip_nighttime: bool = True, **kwargs) -> None:
        """Plot a pie chart of the flag value distribution."""
        counts = self.counts(skip_nighttime=skip_nighttime, normalize=True)
        defaults = {"labels": counts.index, "autopct": "%1.1f%%", "startangle": 90}
        counts.plot.pie(**(defaults | kwargs))

    def plot(self, sdf: SolarDataFrame, **kwargs) -> None:
        """Plot the original data colored by flag values for visual inspection."""
        if not isinstance(self._series, SolarSeries):
            logger.warning("testplot is only valid for SolarSeries. Cannot plot.")
            return

        for _, obj in inspect.getmembers(
            qcrad, predicate=lambda obj: isinstance(obj, helpers.QCTest)
        ):
            if obj.name == self._series.name:
                plot_func = obj._plot_func
                break
        else:
            logger.warning(
                f"No QCTest found for series '{self._series.name}'. Cannot plot."
            )
            return

        return plot_func(sdf, self._series)

    def heatmap(self) -> None:
        """Plot a date-time map of the original data colored by flag values for visual inspection."""
        if not isinstance(self._series, SolarSeries):
            logger.warning("testplot is only valid for SolarSeries. Cannot plot.")
            return

        cmap = ListedColormap([QC_COLOR_FAILED, QC_COLOR_NOT_VERIFIABLE, QC_COLOR_PASSED])
        norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

        fig, ax = plt.subplots(1, 1, figsize=(12, 5), layout="constrained")
        ax.set_facecolor("white")

        kwargs = {"twilight_line": True, "aggfunc": "median", "cmap": cmap, "norm": norm}
        self._series.astype(np.int8).solarplot.heatmap(ax=ax,colorbar=False, **kwargs)
        ax.set_title(f"QC Flag for -- {self._series.name} --")

        mesh = ax.collections[0]
        cax = ax.inset_axes([0., -0.15, 0.4, 0.03], transform=ax.transAxes)
        cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
        cbar.set_ticks([-1, 0, 1])
        cbar.ax.set_xticklabels(["FAILED", "NOT VERIFIABLE", "PASSED"])
