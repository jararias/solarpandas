
from .base import SolarDataFrame


def read_csv(filename, **kwargs):
    """Read a CSV file into a :class:`SolarDataFrame`.

    Parameters
    ----------
    filename : str or pathlib.Path
        Input file path.
    **kwargs : Any
        Additional keyword arguments passed to
        :meth:`SolarDataFrame.read_csv`.

    Returns
    -------
    SolarDataFrame
        Data loaded from ``filename`` with metadata recovered from the first
        JSON-encoded line.

    Examples
    --------
    >>> import solarpandas as sp
    >>> sdf = sp.read_csv("station_data.csv")
    >>> isinstance(sdf, sp.SolarDataFrame)
    True
    """
    return SolarDataFrame.read_csv(filename, **kwargs)


def read_parquet(filename):
    """Read a Parquet file into a :class:`SolarDataFrame`.

    Parameters
    ----------
    filename : str or pathlib.Path
        Input file path.

    Returns
    -------
    SolarDataFrame
        Data loaded from ``filename`` with metadata recovered from the Parquet
        schema metadata.

    Examples
    --------
    >>> import solarpandas as sp
    >>> sdf = sp.read_parquet("station_data.parquet")
    >>> sdf.latitude
    37.0
    """
    return SolarDataFrame.read_parquet(filename)
