
"""Low-level utility functions shared by BSRN readers and parsers."""

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset


def guess_time_resolution(df_or_series):  # , enable_warnings=True):
    """Infer sampling step from a datetime index.

    Parameters
    ----------
    df_or_series : pandas.DataFrame or pandas.Series
        Object with datetime-like index.

    Returns
    -------
    pandas.Timedelta or None
        Inferred time step, or ``None`` when no robust estimate is possible.

    Notes
    -----
    First attempts :func:`pandas.infer_freq`; if unavailable, falls back to the
    minimum observed lag and validates that it can reconstruct the index.
    """

    index = df_or_series.index

    if (inferred_freq := pd.infer_freq(index)) is not None:
        return pd.to_timedelta(to_offset(inferred_freq))

    # for most cases, the following should work. If there are many missings
    # in the series, little can be done...
    step = pd.to_timedelta(np.diff(index.to_numpy()).min())
    reconstructed_index = pd.date_range(index[0], index[-1], freq=step)
    if index.isin(reconstructed_index).all():
        return step

    return None


def time_interpolation(data: pd.DataFrame, new_index: pd.DatetimeIndex):
    """Interpolate data in time over a target index.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data indexed by timestamps.
    new_index : pandas.DatetimeIndex
        Target index to interpolate onto.

    Returns
    -------
    pandas.DataFrame
        Interpolated dataframe aligned to ``new_index``.
    """
    # time interpolation
    extended_index = data.index.append(new_index).sort_values()
    new_data = data.reindex(extended_index).interpolate(method='time', limit=1)
    # drop duplicated indices
    new_data = new_data.reset_index()
    index_name = new_data.index.name or 'index'
    new_data = new_data.drop_duplicates(subset=index_name, keep='first')
    new_data.index = new_data[index_name]
    new_data.drop(columns=index_name, inplace=True)
    new_data = new_data.reindex(new_index)
    return new_data
