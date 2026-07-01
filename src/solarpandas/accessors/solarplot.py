"""Plotting utilities and scales for solar data visualization."""

from typing import Callable, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.dates import DateFormatter
from matplotlib.scale import ScaleBase
from matplotlib.transforms import Transform

from ..base import SolarDataFrame, SolarSeries
from ..helpers import infer_time_step, normalize

logger.disable(__name__)
logger = logger.opt(colors=True)


class _DiurnalMapper:
    """Map matplotlib date numbers to a compressed daytime-only timeline."""

    def __init__(self, real_numdates: np.ndarray, nominal_step_days: float):
        real_numdates = np.asarray(real_numdates, dtype=float)
        if real_numdates.ndim != 1 or len(real_numdates) == 0:
            raise ValueError("diurnal mapper requires at least one timestamp")

        diffs = np.diff(real_numdates, prepend=real_numdates[0])
        clipped_diffs = np.minimum(diffs, nominal_step_days)
        clipped_diffs[0] = 0.0

        self.real = real_numdates
        self.diurnal = real_numdates[0] + np.cumsum(clipped_diffs)

    def to_diurnal(self, values: np.ndarray | float):
        values = np.asarray(values, dtype=float)
        return np.interp(values, self.real, self.diurnal)

    def to_real(self, values: np.ndarray | float):
        values = np.asarray(values, dtype=float)
        return np.interp(values, self.diurnal, self.real)


class _DiurnalTransform(Transform):
    input_dims = output_dims = 1
    has_inverse = True

    def __init__(self, mapper: _DiurnalMapper):
        super().__init__()
        self._mapper = mapper

    def transform_non_affine(self, values):
        return self._mapper.to_diurnal(values)

    def inverted(self):
        return _InvertedDiurnalTransform(self._mapper)


class _InvertedDiurnalTransform(Transform):
    input_dims = output_dims = 1
    has_inverse = True

    def __init__(self, mapper: _DiurnalMapper):
        super().__init__()
        self._mapper = mapper

    def transform_non_affine(self, values):
        return self._mapper.to_real(values)

    def inverted(self):
        return _DiurnalTransform(self._mapper)


class _DiurnalDateLocator(mticker.Locator):
    """Wrap a DateLocator so it operates in real time and returns diurnal ticks."""

    def __init__(self, mapper: _DiurnalMapper, base_locator):
        super().__init__()
        self._mapper = mapper
        self._base = base_locator

    def set_axis(self, axis):
        super().set_axis(axis)
        self._base.set_axis(axis)

    def tick_values(self, vmin, vmax):
        rmin, rmax = self._mapper.to_real(np.array([vmin, vmax], dtype=float))
        if rmin > rmax:
            rmin, rmax = rmax, rmin
        rticks = self._base.tick_values(rmin, rmax)
        return self._mapper.to_diurnal(np.asarray(rticks, dtype=float))

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def view_limits(self, vmin, vmax):
        rmin, rmax = self._mapper.to_real(np.array([vmin, vmax], dtype=float))
        rvmin, rvmax = self._base.view_limits(rmin, rmax)
        dvmin, dvmax = self._mapper.to_diurnal(np.array([rvmin, rvmax], dtype=float))
        return dvmin, dvmax


class _DiurnalDateFormatter(mticker.Formatter):
    """Wrap a DateFormatter-like formatter with diurnal<->real mapping."""

    def __init__(self, mapper: _DiurnalMapper, base_formatter):
        super().__init__()
        self._mapper = mapper
        self._base = base_formatter

    def set_axis(self, axis):
        super().set_axis(axis)
        self._base.set_axis(axis)

    def set_locs(self, locs):
        real_locs = self._mapper.to_real(np.asarray(locs, dtype=float))
        if hasattr(self._base, "set_locs"):
            self._base.set_locs(real_locs)

    def __call__(self, x, pos=None):
        rx = float(self._mapper.to_real(np.array([x], dtype=float))[0])
        return self._base(rx, pos)


class _DiurnalScale(ScaleBase):
    name = "diurnal"

    def __init__(self, axis, *, mapper: _DiurnalMapper, locator=None, formatter=None):
        super().__init__(axis)
        self._mapper = mapper
        self._base_locator = locator or mpl.dates.AutoDateLocator()
        self._base_formatter = formatter or mpl.dates.AutoDateFormatter(
            self._base_locator
        )

    def get_transform(self):
        return _DiurnalTransform(self._mapper)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(_DiurnalDateLocator(self._mapper, self._base_locator))
        axis.set_major_formatter(
            _DiurnalDateFormatter(self._mapper, self._base_formatter)
        )

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return vmin, vmax


if "diurnal" not in mpl.scale.get_scale_names():
    mpl.scale.register_scale(_DiurnalScale)


class SolarPlotAccessor:
    """Accessor with high-level plotting methods for solar time series.

    Methods
    -------
    diurnal
        All-data compressed timeline removing nighttime gaps.
    heatmap
        Date-time heatmap of a single variable.
    rollingday
        Interactive day-by-day time series with keyboard/scroll navigation.

    Examples
    --------
    >>> sdf.solarplot.diurnal(column="ghi")
    >>> sdf.solarplot.heatmap(column="ghi", time_ref="tst")
    >>> sdf.solarplot.rollingday("ghi", window_size=3)
    """

    def __init__(self, sdf_obj):
        self._sdf = self._validate(sdf_obj)

    @staticmethod
    def _validate(obj):
        if not isinstance(obj, (SolarSeries, SolarDataFrame)):
            name = obj.__class__.__name__
            raise AttributeError(
                f"required a SolarSeries or SolarDataFrame instance. Got {name}"
            )
        return obj

    def diurnal(
        self,
        column: str | list[str] | tuple[str, ...] | None = None,
        max_sza: float = 95.0,
        locator=None,
        formatter=None,
        **kwargs,
    ) -> plt.Figure:
        """Plot one or more variables on a compressed daytime-only timeline.

        Parameters
        ----------
        column : str, list[str], tuple[str, ...], or None, default None
            Column(s) to plot for dataframe inputs. Ignored for series inputs.
        max_sza : float, default 95.0
            Maximum solar zenith angle used to define daytime samples.
        locator, formatter : Any, optional
            Optional matplotlib date locator/formatter for x-axis ticks.
        **kwargs : Any
            Extra keyword arguments forwarded to ``Axes.plot``.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the diurnal plot.

        Examples
        --------
        >>> fig = sdf.solarplot.diurnal(column="ghi", color="gold", lw=1.5)
        >>> fig = sdf.solarplot.diurnal(column=["ghi", "dni"], max_sza=90)
        """

        if isinstance(self._sdf, SolarSeries):
            if column is not None:
                logger.warning("Column name(s) ignored when plotting a SolarSeries.")
            columns = [self._sdf.name or "_unnamed_"]
            sdf = self._sdf.to_frame(columns[0])
        else:
            if column is None:
                columns = self._sdf.columns
            elif isinstance(column, str):
                columns = [column]
            elif isinstance(column, (list, tuple)):
                columns = list(column)
            else:
                raise TypeError(
                    "`column` must be a string, a list/tuple of strings, or None."
                )

            missing = [c for c in columns if c not in self._sdf.columns]
            if missing:
                logger.warning(
                    f"Columns {missing} not found in dataframe. Defaulting to the first column."
                )
                columns = [self._sdf.columns[0]]

            sdf = self._sdf[columns]

        sza = self._sdf.solpos.zenith
        daytime_mask = sza < max_sza
        df = sdf.where(sza < 91).loc[daytime_mask, columns].copy()
        if df.empty:
            raise ValueError("No daytime samples available with the selected max_sza.")

        step = pd.to_timedelta(infer_time_step(sdf))
        step_days = step / pd.Timedelta("1D")
        real_numdates = mpl.dates.date2num(df.index.to_pydatetime())
        mapper = _DiurnalMapper(real_numdates, nominal_step_days=float(step_days))

        # plt.style.use("solarpandas-diurnal")
        if (ax := kwargs.pop("ax", None)) is None:
            _, ax = plt.subplots(1, 1, figsize=(12, 6), layout="constrained")

        ax.set_xscale("diurnal", mapper=mapper, locator=locator, formatter=formatter)
        ax.plot(df.index, df, **kwargs)

        return ax.get_figure()

    def heatmap(
        self,
        column: str | None = None,
        time_ref: Literal["lst", "tst", "lat", "utc"] = "tst",
        max_sza: float | None = 90.0,
        colorbar: bool = True,
        colorbar_title: str | None = None,
        twilight_line: bool = False,
        twilight_line_kwargs: dict | None = None,
        aggfunc: str | Callable = "mean",
        **kwargs,
    ) -> plt.Figure:
        """Render a date-time heatmap for a selected variable.

        Parameters
        ----------
        column : str or None, default None
            Column to plot for dataframe inputs. Defaults to first column.
        time_ref : {"lst", "tst", "lat", "utc"}, default "tst"
            Time reference used for the y-axis.
        max_sza : float or None, default 90.0
            Nighttime masking threshold. Use ``None`` to disable masking.
        colorbar : bool, default True
            Whether to add a colorbar.
        twilight_line : bool, default False
            Whether to overlay sunrise and sunset curves.
        aggfunc : str or Callable, default "mean"
            Aggregation function used in the date-time pivot table.
        **kwargs : Any
            Extra keyword arguments forwarded to ``Axes.pcolormesh``.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the heatmap.

        Examples
        --------
        >>> fig = sdf.solarplot.heatmap(column="ghi")
        >>> fig = sdf.solarplot.heatmap(column="dni", time_ref="utc", cmap="plasma")
        """

        MAP_OF_YLABELS = {
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
                logger.warning(
                    "No column specified for plotting. Defaulting to the first column."
                )
                column = self._sdf.columns[0]
            elif column not in self._sdf.columns:
                logger.warning(
                    f"Column '{column}' not found in dataframe. Defaulting to the first column."
                )
                column = self._sdf.columns[0]
            sdf = self._sdf[[column]]

        df = pd.DataFrame(
            sdf.where(self._sdf.solpos.zenith < (max_sza or 180.0), pd.NA)
        )

        # extend the dataframe to have a complete first and last days
        df = normalize(df)

        time_step = infer_time_step(df)

        if time_ref.casefold() == "lst":
            df = df.set_index(self._sdf.solpos.lst)
        elif time_ref.casefold() in ("tst", "lat"):
            df = df.set_index(self._sdf.solpos.tst)
        else:
            df = df.set_index(self._sdf.index.tz_convert("UTC").tz_localize(None))

        if time_ref.casefold() in ("lst", "tst", "lat"):
            df = df.set_index(df.index.round(time_step))

        table = df.assign(date=df.index.date, time=df.index.time).pivot_table(
            index="time", columns="date", values=column, aggfunc=aggfunc
        )

        date_coords = table.columns
        time_coords = table.index.map(lambda t: np.datetime64(time_to_minutes(t), "m"))

        plt.style.use("solarpandas-dtmap")
        if "rc" in kwargs:
            mpl.rcParams.update(kwargs.pop("rc"))

        if (ax := kwargs.pop("ax", None)) is None:
            _, ax = plt.subplots(1, 1, figsize=(12, 6), layout="constrained")

        mesh = ax.pcolormesh(date_coords, time_coords, table.values, **kwargs)
        if colorbar:
            plt.colorbar(
                mesh,
                ax=ax,
                label=colorbar_title or column,
                pad=0.01,
                fraction=0.025,
                shrink=1.0,
            )

        if twilight_line:

            def get_twilight(which: str):
                twilight = (
                    getattr(self._sdf.solpos, which)(units=time_ref)
                    .resample("D")
                    .median()
                    .dt.round("1s")
                    .dt.time.map(lambda t: np.datetime64(time_to_minutes(t), "m"))
                )
                twilight = twilight.set_axis(twilight.index.date)
                return twilight.reindex(
                    date_coords, method="nearest", tolerance=pd.Timedelta("1D")
                )

            default_twilight_kwargs = {"color": "purple", "ls": "--", "lw": 1.5}
            twilight_line_kwargs = default_twilight_kwargs | (
                twilight_line_kwargs or {}
            )
            ax.plot(
                date_coords,
                get_twilight("sunrise"),
                label="Sunrise",
                **twilight_line_kwargs,
            )
            ax.plot(
                date_coords,
                get_twilight("sunset"),
                label="Sunset",
                **twilight_line_kwargs,
            )

        ax.set_xlabel("Date")

        ax.yaxis.set_major_formatter(DateFormatter("%H:%M"))
        ax.set_ylim(np.datetime64(0, "m"), np.datetime64(24 * 60, "m"))
        ax.set_ylabel(MAP_OF_YLABELS.get(time_ref.casefold()))

        return ax.get_figure()

    def rolling(
        self,
        column: str | list[str] | tuple[str, ...] | None = None,
        step: int = 1,
        window_size: int = 1,
        max_sza: float = 95.0,
        y_scale: Literal["per_day", "global"] = "per_day",
        plot_kwargs: dict[str, dict] | None = None,
        **kwargs,
    ) -> plt.Figure:
        """Plot a time series with a rolling window of a given size.

        Parameters
        ----------
        column : str, list[str], tuple[str, ...], or None, default None
            Column(s) to plot for dataframe inputs. Ignored for series inputs.
            When ``None`` all columns are shown.
        step : int, default 1
            Number of days to navigate when using the left/right arrow keys or
            mouse scroll.
        window_size : int, default 1
            Number of consecutive days shown in the plot area at once.
        max_sza : float, default 95.0
            Maximum solar zenith angle of the data to plot. Timestamps with
            SZA above this threshold are excluded.
        y_scale : {"per_day", "global"}, default "per_day"
            Y-axis scaling strategy.
            ``"per_day"`` autoscales the y-axis to the data visible in the
            current window on every navigation step.
            ``"global"`` fixes the y-axis to the range of the full dataset
            and keeps it constant while navigating.
        plot_kwargs : dict[str, dict] or None, default None
            Per-column matplotlib keyword arguments. Keys are column names;
            values are dicts of kwargs forwarded to ``Axes.plot`` for that
            specific line. Columns absent from this dict inherit ``**kwargs``.
        **kwargs : Any
            Global keyword arguments forwarded to ``Axes.plot`` for every
            line. Per-column entries in *plot_kwargs* take precedence.

        Returns
        -------
        matplotlib.figure.Figure
            Interactive figure. Use left/right arrow keys or mouse wheel to
            navigate between days.

        Examples
        --------
        >>> fig = sdf.solarplot.rollingday(
        ...     column=["ghi", "dni"],
        ...     window_size=3,
        ...     plot_kwargs={
        ...         "ghi": {"color": "gold", "lw": 2.0},
        ...         "dni": {"color": "tomato", "ls": "--"},
        ...     },
        ...     lw=1.0,
        ... )
        >>> plt.show()
        """

        if window_size < 1:
            raise ValueError("`window_size` must be >= 1")

        # 1. Resolve columns
        if isinstance(self._sdf, SolarSeries):
            if column is not None:
                logger.warning("Column name(s) ignored when plotting a SolarSeries.")
            columns = [self._sdf.name or "_unnamed_"]
            sdf = self._sdf.to_frame(columns[0])
        else:
            if column is None:
                columns = list(self._sdf.columns)
            elif isinstance(column, str):
                columns = [column]
            elif isinstance(column, (list, tuple)):
                columns = list(column)
            else:
                raise TypeError(
                    "`column` must be a string, a list/tuple of strings, or None."
                )

            missing = [c for c in columns if c not in self._sdf.columns]
            if missing:
                logger.warning(
                    f"Columns {missing} not found in dataframe. Skipping them."
                )
                columns = [c for c in columns if c in self._sdf.columns]
            if not columns:
                raise ValueError("No valid columns to plot.")
            sdf = self._sdf[columns]

        # 2. Pre-compute SZA and time step
        sza = self._sdf.solpos.zenith
        step_days = float(pd.to_timedelta(infer_time_step(sdf)) / pd.Timedelta("1D"))

        # in order to show entire "solar" days I need to set the time index in true
        # solar time coordinates. Otherwise, data from sites far from the prime meridian
        # would have their solar days split across two calendar days
        sdf = sdf.set_index(sdf.solpos.tst).copy()
        sza.index = (
            sdf.index
        )  # apply the same tst index to sza for easier masking later

        # 3. Calendar dates for navigation — tz-independent via .date property
        all_dates = np.unique(sdf.index.date)  # sorted array of datetime.date
        n_total = len(all_dates)

        # 4. Figure setup
        if (ax := kwargs.pop("ax", None)) is None:
            _, ax = plt.subplots(1, 1, figsize=(12, 5), layout="constrained")

        per_col = plot_kwargs or {}

        # Pre-compute global y-limits (physical daytime only)
        global_ylim = None
        if y_scale == "global":
            all_day = sdf.where(sza < 93)
            vmin_g, vmax_g = all_day.min().min(), all_day.max().max()
            if pd.notna(vmin_g) and pd.notna(vmax_g) and vmin_g < vmax_g:
                pad_g = (vmax_g - vmin_g) * 0.05
                global_ylim = (vmin_g - pad_g, vmax_g + pad_g)

        # 5. Navigation state and draw function
        state = {"idx": 0}

        def _update(i: int) -> None:
            # define viewing window limits...
            state["idx"] = max(0, min(i, n_total - window_size))
            idx_start = state["idx"]
            idx_end = min(idx_start + window_size - 1, n_total - 1)

            # Select timestamps: within window dates AND below max_sza
            window_dates = all_dates[idx_start : idx_end + 1]
            date_mask = np.isin(sdf.index.date, window_dates)
            sza_mask = (sza < max_sza).values
            win_mask = date_mask & sza_mask

            sdf_win = sdf.loc[win_mask]
            sza_win = sza.loc[win_mask]
            sdf_day = sdf_win.where(sza_win < 91)

            ax.cla()

            if len(sdf_win) > 0:
                real_nums = mpl.dates.date2num(sdf_win.index.to_pydatetime())
                mapper = _DiurnalMapper(real_nums, nominal_step_days=step_days)
                ax.set_xscale("diurnal", mapper=mapper)
                for col in columns:
                    kw = {**kwargs, **per_col.get(col, {})}
                    ax.plot(sdf_day.index, sdf_day[col], label=col, **kw)
                if len(columns) > 1:
                    ax.legend()
                ax.autoscale_view()

            if global_ylim is not None:
                ax.set_ylim(*global_ylim)
            elif y_scale == "per_day" and not sdf_day.empty:
                vmin = sdf_day.min().min()
                vmax = sdf_day.max().max()
                if pd.notna(vmin) and pd.notna(vmax) and vmin < vmax:
                    pad = (vmax - vmin) * 0.05
                    ax.set_ylim(vmin - pad, vmax + pad)

            date_start = all_dates[idx_start]
            date_end = all_dates[idx_end]
            date_str = (
                str(date_start)
                if window_size == 1
                else f"{date_start} \u2013 {date_end}"
            )
            ax.set_title(
                f"{date_str}  [{idx_start + 1}/{n_total}]"
                "  \u2190\u2192 or scroll to navigate"
            )
            ax.get_figure().canvas.draw_idle()

        def _on_key(event) -> None:
            if (
                step_ := step
                if event.key == "right"
                else -step
                if event.key == "left"
                else 0
            ):
                _update(state["idx"] + step_)

        def _on_scroll(event) -> None:
            _update(state["idx"] + (step if event.step < 0 else -step))

        # 6. Connect events and show first window
        fig = ax.get_figure()

        _non_interactive = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
        if mpl.get_backend().lower() in _non_interactive:
            logger.warning(
                f"Backend '{mpl.get_backend()}' is non-interactive; "
                "keyboard/scroll navigation will not work."
            )

        fig.canvas.mpl_connect("key_press_event", _on_key)
        fig.canvas.mpl_connect("scroll_event", _on_scroll)

        _update(0)

        return fig
