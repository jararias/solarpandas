
"""Accessor API to run qcrad quality-control checks on solar data."""

from functools import lru_cache, reduce
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.colors import BoundaryNorm, ListedColormap

from ..base import SolarDataFrame, SolarSeries
from ..helpers import infer_time_step
from ..mplstyles import QC_COLOR_FAILED, QC_COLOR_PASSED
from ..qcontrol import qcrad

logger.disable(__name__)
logger = logger.opt(colors=True)


# The dataframes of pandas are, by design, mutable and unhashable. To be able
# to cache QC results based on the content of the dataframe, we need a hashable
# wrapper that computes a hash based on the content of the dataframe. This is
# what HashableDF does. It computes a hash based on the content of the dataframe
# (including index) and allows us to use it as a key for caching QC results.
class HashableDF:
    """Hashable wrapper for dataframe content used by the QC cache."""

    def __init__(self, unhashable_df: SolarDataFrame | pd.DataFrame):
        self.dataframe = unhashable_df
        # pd.util.has_pandas_object devuelve un array de hashes para cada fila,
        # sumamos para obtener un hash que representa el contenido de todo el DataFrame
        self._hash = int(pd.util.hash_pandas_object(self.dataframe, index=True).sum())

    def __hash__(self):
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HashableDF):
            return False
        return self.dataframe.equals(other.dataframe)


@lru_cache(maxsize=None)
def _run_cached_qc(hashdf: HashableDF) -> SolarDataFrame:
    """Run all qcrad tests and cache the resulting QC dataframe."""
    logger.debug("performing cached quality control...")

    sdf = hashdf.dataframe

    tests = []
    tests.append(qcrad.ghi_ppl(sdf))
    tests.append(qcrad.dif_ppl(sdf))
    tests.append(qcrad.dni_ppl(sdf))
    tests.append(qcrad.ghi_erl(sdf))
    tests.append(qcrad.dif_erl(sdf))
    tests.append(qcrad.dni_erl(sdf))
    tests.append(qcrad.Kn_ppl(sdf))
    tests.append(qcrad.Kn_erl(sdf))
    tests.append(qcrad.KT_erl(sdf))
    tests.append(qcrad.K_erl(sdf))
    tests.append(qcrad.K_erl_clear(sdf))
    tests.append(qcrad.closure(sdf))
    tests.append(qcrad.trackeroff(sdf))

    return pd.concat(tests, axis=1)


def clear_qc_cache() -> None:
    """Clear the in-memory quality-control cache.

    Examples
    --------
    >>> import solarpandas as sp
    >>> sp.clear_qc_cache()
    """
    _run_cached_qc.cache_clear()
    logger.debug("qc cache cleared")


def get_qc_cache_info():
    """Return cache statistics for quality-control computations.

    Returns
    -------
    dict[str, int | None]
        Dictionary with ``hits``, ``misses``, ``current_size`` and ``max_size``.

    Examples
    --------
    >>> import solarpandas as sp
    >>> info = sp.get_qc_cache_info()
    >>> "misses" in info
    True
    """
    info = _run_cached_qc.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "current_size": info.currsize,
        "max_size": info.maxsize,
    }


_COMPONENT_TO_TEST_MAP = {
    "ghi": {
        "1-component": ["ghi_ppl", "ghi_erl"],
        "2-component": ["Kn_ppl", "KT_erl", "K_erl", "K_erl_clear", "trackeroff"],
        "3-component": ["closure"]
    },
    "dni": {
        "1-component": ["dni_ppl", "dni_erl"],
        "2-component": ["Kn_ppl", "Kn_erl", "K_erl", "K_erl_clear", "trackeroff"],
        "3-component": ["closure"]
    },
    "dif": {
        "1-component": ["dif_ppl", "dif_erl"],
        "2-component": [],
        "3-component": ["closure"]
    }
}


@pd.api.extensions.register_dataframe_accessor("qc")
class QualityControlAccessor:
    """Accessor to run and query qcrad quality-control flags.

    Examples
    --------
    >>> qc = sdf.qc
    >>> qc.tests.columns
    >>> qc.failed(component="ghi")
    """

    def __init__(self, sdf_obj):
        self._sdf = self._validate(sdf_obj)
        self._tests = _run_cached_qc(HashableDF(self._sdf))

    @staticmethod
    def _validate(obj):
        if not isinstance(obj, SolarDataFrame):
            name = obj.__class__.__name__
            raise AttributeError(f"required a SolarDataFrame instance. Got {name}")
        time_step = infer_time_step(obj)
        if time_step is None:
            logger.warning(
                "Could not infer the time step of the data. QC tests may be inaccurate or fail "
                "if it is other than one minute. Please, make sure the index is a DateTimeIndex "
                "with a regular 1-minute frequency.")
        elif time_step != pd.Timedelta("1min"):
            logger.warning(
                f"the inferred time step is {time_step}. Please, be aware that QC tests are "
                "designed for 1-minute data and may not be accurate or fail.")
        return obj

    def __getitem__(self, key: str) -> SolarSeries:
        """Return one QC test series by its column name."""
        if key not in self._tests.columns:
            raise KeyError(f"QC test '{key}' not found in results.")
        return self._tests[key]

    def __getattr__(self, name: str) -> SolarSeries:
        """Access QC tests as attributes when names match columns."""
        if name not in self._tests.columns:
            raise AttributeError(f"QC test '{name}' not found in results.")
        return self._tests[name]

    @property
    def tests(self) -> SolarDataFrame:
        """Return the full QC test result dataframe."""
        return self._tests

    def filter(
        self,
        component: Literal["ghi", "dni", "dif"] | None = None,
        *,
        tests: list[str] | None = None,
        like: str | None = None,
        regex: str | None = None,
    ) -> pd.DataFrame:
        """Filter QC tests by component, explicit names, or pattern.

        Parameters
        ----------
        component : {"ghi", "dni", "dif"} or None, default None
            Convenience selector for pre-defined test groups.
        tests : list[str] or None, default None
            Explicit test names.
        like : str or None, default None
            Substring pattern forwarded to ``DataFrame.filter``.
        regex : str or None, default None
            Regex pattern forwarded to ``DataFrame.filter``.

        Returns
        -------
        pandas.DataFrame
            Subset of QC test columns.
        """

        if component is not None:
            if component.casefold() not in ("ghi", "dni", "dif"):
                raise ValueError("component must be one of 'ghi', 'dni', 'dif' or None")

            if any([tests, like, regex]):
                logger.warning("Cannot specify `component` together with `tests`, `like` or `regex` "
                               "filters. Ignoring filters and using component only.")

            logger.debug(f"Filtering QC tests for component '{component}'")
            tests = reduce(lambda x, y: x + y, _COMPONENT_TO_TEST_MAP.get(component).values())
            logger.debug(f"Tests for component '{component}': {tests}")

        if tests is None and like is None and regex is None:
            return self._tests
        tests = self._tests.filter(items=tests, like=like, regex=regex, axis=1)
        logger.info(f"Filtered QC tests: {tests.columns.tolist()}")
        return tests

    def failed(
        self,
        component: Literal["ghi", "dni", "dif"] | None = None,
        *,
        tests: list[str] | None = None,
        like: str | None = None,
        regex: str | None = None,
    ) -> pd.Series:
        """Return a boolean mask where at least one selected test fails."""

        return (
            self.filter(component, tests=tests, like=like, regex=regex)
            .apply(lambda test: test.flag.fails)
            .any(axis=1)
        )

    def passed(
        self,
        component: Literal["ghi", "dni", "dif"] | None = None,
        *,
        tests: list[str] | None = None,
        like: str | None = None,
        regex: str | None = None,
    ) -> pd.Series:
        """Return a boolean mask where all selected tests pass or are neutral."""

        return (
            self.filter(component, tests=tests, like=like, regex=regex)
            .apply(lambda test: test.flag.passes | test.flag.not_verifiable)
            .all(axis=1)
        )

    def mask_failed(
        self,
        component: Literal["ghi", "dni", "dif"] | None = None,
        *,
        tests: list[str] | None = None,
        like: str | None = None,
        regex: str | None = None,
        **kwargs
    ) -> pd.DataFrame:
        """Mask original values where selected QC tests fail.

        Returns
        -------
        pandas.DataFrame
            Copy of the original data with failed timestamps masked.
        """

        failed = self.failed(component, tests=tests, like=like, regex=regex)

        if component is None:
            return self._sdf.mask(failed, **kwargs)

        masked_sdf =self._sdf.copy()
        masked_sdf[component] = masked_sdf[component].mask(failed, **kwargs)
        return masked_sdf

    def heatmap(
        self,
        component: Literal["ghi", "dni", "dif"] | None = None,
        *,
        tests: list[str] | None = None,
        like: str | None = None,
        regex: str | None = None,
        combined: bool = False,
        **kwargs
    ) -> plt.Figure:
        """Render a QC pass/fail heatmap over time.

        Parameters
        ----------
        combined : bool, default False
            If ``True``, encodes failure severity by component groups.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the heatmap.
        """

        if not combined:
            series = self.passed(component, tests=tests, like=like, regex=regex).astype(np.int8)
            cmap = ListedColormap([QC_COLOR_FAILED, QC_COLOR_PASSED])
            norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
            ticks = {0: "FAILED", 1: "PASSED"}
            cax_bounds = [0.05, -0.15, 0.3, 0.03]
        else:

            def get_failed(components):
                return self.failed(tests=_COMPONENT_TO_TEST_MAP.get(component).get(components))

            series = self._sdf.replace_data(other=0.).iloc[:, 0].astype(np.int8)
            series.loc[get_failed("1-component")] = np.int8(1)
            series.loc[get_failed("2-component")] = np.int8(2)
            series.loc[get_failed("3-component")] = np.int8(3)

            cmap = ListedColormap(["#e6f2ff", "#84e184", "#4d94ff", "#ff6666"])
            norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
            ticks = {0: "Passed", 1: "1-comp", 2: "2-comp", 3: "3-comp"}
            cax_bounds = [0.025, -0.15, 0.35, 0.03]

        title = "QC Results"
        if component is not None:
            title += f" for {component.upper()}"
        network = self._sdf.custom_metadata.get("network", None)
        if network is not None and network.casefold() == "bsrn":
            station = self._sdf.custom_metadata.get("station", "unknown station")
            location = self._sdf.custom_metadata.get("location", "unknown location")
            acronym = self._sdf.custom_metadata.get("acronym", "unknown acronym")
            title += f" at {station}, {location} ({acronym.upper()}, BSRN)"
        title += f" (lat={self._sdf.latitude:.4f}, lon={self._sdf.longitude:.4f}, alt={self._sdf.elevation:.0f} m)"

        fig, ax = plt.subplots(1, 1, figsize=(14, 5), layout="constrained")
        ax.set_facecolor("white")

        kwargs = {"twilight_line": True, "aggfunc": "median", "cmap": cmap, "norm": norm}
        series.solarplot.heatmap(ax=ax, colorbar=False, **kwargs)
        ax.set_title(title)

        mesh = ax.collections[0]
        cax = ax.inset_axes(cax_bounds, transform=ax.transAxes)
        cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
        cbar.set_ticks(list(ticks.keys()))
        cbar.ax.set_xticklabels(list(ticks.values()))
