from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd
from loguru import logger
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
    register_extension_dtype,
    register_series_accessor,
)

from ..base import SolarSeries

logger.disable(__name__)
logger = logger.opt(colors=True)


class QCFlagEnum(IntEnum):
    FAILED = np.int8(-1)
    NOT_VERIFIABLE = np.int8(0)
    PASSED = np.int8(1)

    @classmethod
    def values(cls) -> list[int]:
        """Return all valid flag values as a list of ints."""
        return [e.value for e in cls]


_VALID_VALUES = np.array(QCFlagEnum.values(), dtype=np.int8)
_NA_SENTINEL = np.iinfo(np.int8).min  # -128, used as NA


@register_extension_dtype
class FlagDtype(ExtensionDtype):
    """Dtype for QC flag arrays.

    Valid values: -1 (fail), 0 (not verifiable), 1 (passed).
    NA is represented internally as -128.
    """

    name = "QCflag"
    type = np.int8
    na_value = pd.NA

    @classmethod
    def construct_array_type(cls) -> type[FlagArray]:
        return FlagArray

    def __repr__(self) -> str:
        return "FlagDtype()"


class FlagArray(ExtensionArray):
    """ExtensionArray for QC flag values: -1, 0, 1 (or NA).

    Internally stored as int8 with -128 as a sentinel for NA.

    Properties
    ----------
    fails : ndarray[bool]
        True where the flag value is -1.
    passes : ndarray[bool]
        True where the flag value is 1.
    not_verifiable : ndarray[bool]
        True where the flag value is 0.
    """

    dtype = FlagDtype()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def __init__(self, values: np.ndarray, copy: bool = False) -> None:
        """
        Parameters
        ----------
        values : np.ndarray of int8
            Raw storage array. Use _from_sequence for public construction.
        """
        if not isinstance(values, np.ndarray) or values.dtype != np.int8:
            raise TypeError("values must be a np.ndarray of dtype int8")
        self._data = values.copy() if copy else values

    @classmethod
    def _from_sequence(
        cls,
        scalars,
        *,
        dtype=None,
        copy: bool = False,
    ) -> FlagArray:
        """Construct from a sequence of scalars (-1, 0, 1 or NA/None/np.nan)."""
        if dtype is not None and not isinstance(dtype, FlagDtype):
            raise TypeError(f"Cannot construct FlagArray with dtype {dtype!r}")

        obj = np.asarray(scalars, dtype=object)
        na_mask = pd.isna(obj)  # handles None, np.nan and pd.NA vectorially

        raw = np.full(len(obj), _NA_SENTINEL, dtype=np.int8)
        if (~na_mask).any():
            raw[~na_mask] = obj[~na_mask].astype(np.int8)
            invalid = ~na_mask & ~np.isin(raw, _VALID_VALUES)
            if invalid.any():
                bad = np.unique(raw[invalid]).tolist()
                raise ValueError(f"FlagArray only accepts -1, 0 or 1; got {bad}")

        return cls(raw, copy=copy)

    @classmethod
    def _from_sequence_of_strings(
        cls,
        strings,
        *,
        dtype=None,
        copy: bool = False,
    ) -> FlagArray:
        return cls._from_sequence(
            [int(s) if s not in ("", "NA", "<NA>", "nan") else pd.NA for s in strings]
        )

    @classmethod
    def _from_factorized(cls, values: np.ndarray, original: FlagArray) -> FlagArray:
        return cls._from_sequence(values)

    # ------------------------------------------------------------------
    # Required abstract methods
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        result = self._data[key]
        if np.ndim(result) == 0:
            # scalar extraction
            v = int(result)
            return pd.NA if v == _NA_SENTINEL else v
        return type(self)(result)

    def __setitem__(self, key, value) -> None:
        if isinstance(value, type(self)):
            self._data[key] = value._data
            return
        # np.isscalar returns False for pd.NA, None and np.nan, so check explicitly
        if np.isscalar(value) or value is pd.NA or value is None:
            scalars = [value]
        else:
            scalars = np.asarray(value)
        tmp = type(self)._from_sequence(scalars)
        self._data[key] = tmp._data

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other):
        if isinstance(other, FlagArray):
            return self._data == other._data
        return self._data == np.int8(other)

    def isna(self) -> np.ndarray:
        return self._data == _NA_SENTINEL

    def take(
        self,
        indices: np.ndarray,
        *,
        allow_fill: bool = False,
        fill_value=None,
    ) -> FlagArray:
        from pandas.core.algorithms import take

        if allow_fill:
            fill = np.int8(_NA_SENTINEL)
        else:
            fill = None

        result = take(self._data, indices, allow_fill=allow_fill, fill_value=fill)
        return type(self)(result)

    def copy(self) -> FlagArray:
        return type(self)(self._data.copy())

    @classmethod
    def _concat_same_type(cls, to_concat) -> FlagArray:
        return cls(np.concatenate([arr._data for arr in to_concat]))

    # ------------------------------------------------------------------
    # Optional but useful overrides
    # ------------------------------------------------------------------

    @property
    def nbytes(self) -> int:
        return self._data.nbytes

    def _values_for_factorize(self):
        return self._data.astype(object), _NA_SENTINEL

    def _formatter(self, boxed: bool = False):
        def fmt(x):
            return "<NA>" if x is pd.NA else str(x)

        return fmt

    def __repr__(self) -> str:
        return f"FlagArray({self._data.tolist()})"

    # ------------------------------------------------------------------
    # Flag semantics
    # ------------------------------------------------------------------

    @property
    def fails(self) -> np.ndarray:
        """Boolean mask: True where the flag is -1 (failed)."""
        return (self._data == np.int8(-1)) & ~self.isna()

    @property
    def passes(self) -> np.ndarray:
        """Boolean mask: True where the flag is 1 (passed)."""
        return (self._data == np.int8(1)) & ~self.isna()

    @property
    def not_verifiable(self) -> np.ndarray:
        """Boolean mask: True where the flag is 0 (not verifiable)."""
        return (self._data == np.int8(0)) & ~self.isna()


@register_series_accessor("flag")
class FlagAccessor:
    """Accessor for Series with QCflag dtype.

    Usage
    -----
    s.flag.fails           # boolean Series: True where value is -1
    s.flag.passes          # boolean Series: True where value is 1
    s.flag.not_verifiable  # boolean Series: True where value is 0
    """

    def __init__(self, series: pd.Series | SolarSeries) -> None:
        if not isinstance(series.dtype, FlagDtype):
            raise AttributeError(
                "The .flag accessor is only available for Series with dtype 'QCflag'."
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
