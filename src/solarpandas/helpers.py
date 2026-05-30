
"""General helper functions shared across the solarpandas package."""

import pandas as pd
from loguru import logger

logger.disable(__name__)
logger = logger.opt(colors=True)


# TODO: FUNCION PARA RELLENAR HUECOS


def infer_time_step(df_or_s: pd.DataFrame | pd.Series) -> pd.Timedelta | None:
    """Infer the sampling time step from a datetime-like index.

    Parameters
    ----------
    df_or_s : pandas.DataFrame or pandas.Series
        Object with a monotonic datetime-like index.

    Returns
    -------
    pandas.Timedelta or None
        Inferred step. Returns ``None`` if the frequency cannot be inferred and
        no valid time differences are available.

    Notes
    -----
    The function first tries the index ``freq`` attribute and
    :func:`pandas.infer_freq`. If that fails, it falls back to the smallest
    observed lag in ``index.diff()``.
    """
    if (freq := (df_or_s.index.freq or pd.infer_freq(df_or_s.index))) is None:
        logger.warning("Could not infer the index frequency using `pd.infer_freq`")
        time_step = df_or_s.index.diff().unique().drop(pd.NaT, errors="ignore")
        if len(time_step) == 0:
            logger.warning("Could not infer the index time step from the shortest lag between consecutive rows")
            logger.error("No valid time steps found.")
            return None
        return time_step.min()
    return pd.to_timedelta(pd.tseries.frequencies.to_offset(freq))


def normalize(df_or_s: pd.DataFrame | pd.Series, **kwargs) -> pd.DataFrame | pd.Series:
    """Reindex data to complete first and last calendar days.

    Parameters
    ----------
    df_or_s : pandas.DataFrame or pandas.Series
        Input object indexed by timestamps.
    **kwargs : Any
        Extra keyword arguments forwarded to ``DataFrame.reindex`` or
        ``Series.reindex`` (for example ``method='nearest'`` or
        ``fill_value=0``).

    Returns
    -------
    pandas.DataFrame or pandas.Series
        Reindexed object spanning complete days from the first day start to the
        day after the last timestamp (left-inclusive).

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series([1, 2], index=pd.to_datetime(["2024-01-01 12:00", "2024-01-01 13:00"]))
    >>> out = normalize(s)
    >>> out.index.min().hour
    0
    """

    # determine the dataframe or series index frequency and time step
    time_step = infer_time_step(df_or_s)

    # determine the start of the new index that have a complete first day
    start = df_or_s.index.min()
    midnight_start = start.floor("D")
    lag = (start - midnight_start) % time_step
    new_start = midnight_start + lag

    # determine the timestamp ending for pd.date_range
    end = df_or_s.index.max()
    midnight_end = end.floor("D") + pd.Timedelta(days=1)

    new_index = pd.date_range(start=new_start, end=midnight_end, freq=time_step, inclusive="left")
    return df_or_s.reindex(new_index, **kwargs)