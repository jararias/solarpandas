"""Core data containers with site metadata for solar time series.

This module defines :class:`SolarSeries` and :class:`SolarDataFrame`, two pandas
subclasses that keep site-level metadata (latitude, longitude, elevation and
custom metadata) attached to the object through common pandas operations.

The module also provides top-level convenience readers for the custom CSV and
Parquet formats implemented by :class:`SolarDataFrame`.
"""

import copy
import json
import linecache
from numbers import Number
from pathlib import Path
from typing import Self, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from .types import Elevation, Latitude, Longitude, validate_type

logger.enable(__name__)


def _epilogue(obj):
    station = obj.custom_metadata.get("station", "<unknown>")
    epilogue = f"\n[site={station}"
    if (network := obj.custom_metadata.get("network", None)) is not None:
        epilogue += f"/{network}"
    epilogue += (f" latitude={obj.latitude:.4f}\u00b0"
                 f" longitude={obj.longitude:.4f}\u00b0"
                 f" elevation={obj.elevation:.1f} m]")
    return epilogue


class SolarSeries(pd.Series):
    """Solar data series carrying site metadata.

    Parameters
    ----------
    latitude : float
        Site latitude in decimal degrees. Must satisfy ``-90 < lat < 90``.
    longitude : float
        Site longitude in decimal degrees. Must satisfy ``-180 <= lon < 180``.
    elevation : float, default 0.0
        Site elevation in meters.
    custom_metadata : dict or None, default None
        Additional user metadata to attach to the series.

    Notes
    -----
    Metadata are propagated through the custom pandas constructors.
    ``latitude``, ``longitude`` and ``elevation`` are reserved metadata keys.
    They are managed internally and cannot be provided in ``custom_metadata``.
    """
    _metadata = ["_latitude", "_longitude", "_elevation", "_custom_metadata"]

    @property
    def _constructor(self):
        def inner(*args, **kwargs):
            kwargs["latitude"] = self._latitude
            kwargs["longitude"] = self._longitude
            kwargs["elevation"] = self._elevation
            kwargs["custom_metadata"] = self._custom_metadata
            return SolarSeries(*args, **kwargs)

        return inner

    @property
    def _constructor_expanddim(self):
        def inner(*args, **kwargs):
            kwargs["latitude"] = self._latitude
            kwargs["longitude"] = self._longitude
            kwargs["elevation"] = self._elevation
            kwargs["custom_metadata"] = self._custom_metadata
            return SolarDataFrame(*args, **kwargs)

        return inner

    def __init__(
        self,
        *args,
        latitude: Latitude,
        longitude: Longitude,
        elevation: Elevation = 0.0,
        custom_metadata: dict | None = None,
        **kwargs,
    ):
        self._latitude = validate_type(latitude, Latitude)
        self._longitude = validate_type(longitude, Longitude)
        self._elevation = validate_type(elevation, Elevation)
        self._custom_metadata = custom_metadata or {}
        if "latitude" in self._custom_metadata:
            raise ValueError("`latitude` cannot be a key in metadata")
        if "longitude" in self._custom_metadata:
            raise ValueError("`longitue` cannot be a key in metadata")
        if "elevation" in self._custom_metadata:
            raise ValueError("`elevation` cannot be a key in metadata")
        super().__init__(*args, **kwargs)

    @property
    def latitude(self):
        return self._latitude

    @property
    def longitude(self):
        return self._longitude

    @property
    def elevation(self):
        return self._elevation

    @property
    def custom_metadata(self):
        return self._custom_metadata

    def replace_data(
        self, other: pd.Series | pd.DataFrame | Sequence[Number] | Number
    ) -> Self:
        """Create a copy with identical site metadata and new data values.

        Parameters
        ----------
        other : pandas.Series, pandas.DataFrame, sequence, or scalar number
            New data values.

        Returns
        -------
        Self
            A new :class:`SolarSeries` preserving index (when applicable) and
            metadata from the source object.
        """
        kwargs = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "elevation": self.elevation,
            "custom_metadata": copy.deepcopy(self.custom_metadata),
        }
        if isinstance(other, Number):
            return self.__class__(
                data=np.full((len(self),), other), index=self.index, **kwargs
            )
        if isinstance(other, (np.ndarray, list)):
            return self.__class__(data=copy.copy(other), index=self.index, **kwargs)
        return self.__class__(data=copy.copy(other), **kwargs)

    def __repr__(self):
        return super().__repr__() + _epilogue(self)

    def __str__(self):
        return super().__str__()


class SolarDataFrame(pd.DataFrame):
    """Solar dataframe carrying site metadata.

    Parameters
    ----------
    latitude : float
        Site latitude in decimal degrees. Must satisfy ``-90 < lat < 90``.
    longitude : float
        Site longitude in decimal degrees. Must satisfy ``-180 <= lon < 180``.
    elevation : float, default 0.0
        Site elevation in meters.
    custom_metadata : dict or None, default None
        Additional user metadata to keep together with the dataframe.

    Notes
    -----
    Metadata are propagated through the custom pandas constructors.
    ``latitude``, ``longitude`` and ``elevation`` are reserved metadata keys.
    They are managed internally and cannot be provided in ``custom_metadata``.
    """
    _metadata = ["_latitude", "_longitude", "_elevation", "_custom_metadata"]

    @property
    def _constructor(self):
        def inner(*args, **kwargs):
            kwargs["latitude"] = self._latitude
            kwargs["longitude"] = self._longitude
            kwargs["elevation"] = self._elevation
            kwargs["custom_metadata"] = self._custom_metadata
            return SolarDataFrame(*args, **kwargs)

        return inner

    @property
    def _constructor_sliced(self):
        def inner(*args, **kwargs):
            kwargs["latitude"] = self._latitude
            kwargs["longitude"] = self._longitude
            kwargs["elevation"] = self._elevation
            kwargs["custom_metadata"] = self._custom_metadata
            return SolarSeries(*args, **kwargs)

        return inner

    def __init__(
        self,
        *args,
        latitude: Latitude,
        longitude: Longitude,
        elevation: Elevation = 0.0,
        custom_metadata: dict | None = None,
        **kwargs,
    ):
        self._latitude = validate_type(latitude, Latitude)
        self._longitude = validate_type(longitude, Longitude)
        self._elevation = validate_type(elevation, Elevation)
        self._custom_metadata = custom_metadata or {}
        if "latitude" in self._custom_metadata:
            raise ValueError("`latitude` cannot be a key in metadata")
        if "longitude" in self._custom_metadata:
            raise ValueError("`longitue` cannot be a key in metadata")
        if "elevation" in self._custom_metadata:
            raise ValueError("`elevation` cannot be a key in metadata")
        super().__init__(*args, **kwargs)

    @property
    def latitude(self):
        return self._latitude

    @property
    def longitude(self):
        return self._longitude

    @property
    def elevation(self):
        return self._elevation

    @property
    def custom_metadata(self):
        return self._custom_metadata

    def as_pandas(self):
        """Return a plain pandas DataFrame view of this object.

        Returns
        -------
        pandas.DataFrame
            Equivalent dataframe without solarpandas subclass semantics.
        """
        return pd.DataFrame(self)

    def describe(self):
        """Compute descriptive statistics as a plain pandas dataframe.

        Returns
        -------
        pandas.DataFrame
            Result of ``pandas.DataFrame.describe`` on this dataset.
        """
        return self.as_pandas().describe()

    def replace_data(
        self, other: pd.Series | pd.DataFrame | Sequence[Number] | Number
    ) -> Self:
        """Create a copy with identical metadata and replaced data.

        Parameters
        ----------
        other : pandas.Series, pandas.DataFrame, sequence, or scalar number
            New data used to build the cloned object.

        Returns
        -------
        Self
            A new :class:`SolarDataFrame` preserving metadata from the current
            object.
        """
        kwargs = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "elevation": self.elevation,
            "custom_metadata": copy.deepcopy(self.custom_metadata),
        }
        if isinstance(other, Number):
            return self.__class__(
                data=np.full((len(self),), other), index=self.index, **kwargs
            )
        if isinstance(other, (np.ndarray, list)):
            return self.__class__(data=copy.copy(other), index=self.index, **kwargs)
        return self.__class__(data=copy.copy(other), **kwargs)

    def __repr__(self):
        return super().__repr__() + _epilogue(self)

    def __str__(self):
        return super().__str__()

    def to_csv(self, path: str | Path, **kwargs):
        """Write dataframe and metadata to a CSV file.

        The first output line stores a JSON document with metadata. Data values
        are written from the second line onward using ``pandas.DataFrame.to_csv``.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination path.
        **kwargs : Any
            Extra keyword arguments passed to ``DataFrame.to_csv``.
        """
        metadata = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "elevation": self.elevation,
        } | self.custom_metadata

        if not (p := Path(path)).parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)

        default_kwargs = {"header": True}
        with p.open("w") as f:
            f.write(json.dumps(metadata) + "\n")
            pd.DataFrame(self).to_csv(f, **(default_kwargs | kwargs))

    def to_parquet(self, path: str | Path, **kwargs):
        """Write dataframe and metadata to a Parquet file.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination path.
        **kwargs : Any
            Extra keyword arguments passed to ``pyarrow.parquet.write_table``.

        Notes
        -----
        Metadata is stored in the Parquet schema metadata.
        """
        metadata = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "elevation": self.elevation,
        } | self.custom_metadata

        # convert the dataframe to a Arrow Table
        table = pa.Table.from_pandas(self)

        # copy the metadata to a dictionary and convert the numpy
        # arrays and dataframes to a json-serializable format
        df_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                df_metadata[f"{key}/ndarray"] = value.tolist()
            elif isinstance(value, pd.DataFrame):
                df_metadata[f"{key}/dataframe"] = value.to_dict()
            else:
                df_metadata[key] = value

        # add the dictionary to the schema metadata of table. Note
        # that I am using the keyword "syngena" and encode to bytes
        combined_metadata = {
            "solarpandas".encode(): json.dumps(df_metadata).encode(),
            **table.schema.metadata,
        }  # DataFrame"s metadata
        table = table.replace_schema_metadata(combined_metadata)

        # serialize to a parquet file
        p = Path(path)
        if not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        pa.parquet.write_table(table, p, **kwargs)

    @classmethod
    def read_csv(cls, path: str | Path, **kwargs):
        """Read a CSV file written by :meth:`to_csv`.

        Parameters
        ----------
        path : str or pathlib.Path
            Input file path.
        **kwargs : Any
            Additional keyword arguments passed to :func:`pandas.read_csv`.

        Returns
        -------
        SolarDataFrame
            Parsed dataset with restored site metadata.

        Examples
        --------
        >>> sdf.to_csv("data.csv")
        >>> restored = SolarDataFrame.read_csv("data.csv")
        >>> restored.latitude == sdf.latitude
        True
        """
        if not (p := Path(path)).exists():
            raise ValueError(f"missing file {path}")

        metadata_line = linecache.getline(p.as_posix(), 1)
        must_kwargs = {"header": 1, "index_col": 0, "parse_dates": True}
        data = pd.read_csv(path, **(kwargs | must_kwargs))
        metadata = json.loads(metadata_line)

        return cls(
            data=data,
            latitude=float(metadata.pop("latitude")),
            longitude=float(metadata.pop("longitude")),
            elevation=float(metadata.pop("elevation")),
            custom_metadata=metadata,
        )

    @classmethod
    def read_parquet(cls, path: str | Path):
        """Read a Parquet file written by :meth:`to_parquet`.

        Parameters
        ----------
        path : str or pathlib.Path
            Input file path.

        Returns
        -------
        SolarDataFrame
            Parsed dataset with restored site metadata.

        Examples
        --------
        >>> sdf.to_parquet("data.parquet")
        >>> restored = SolarDataFrame.read_parquet("data.parquet")
        >>> restored.longitude == sdf.longitude
        True
        """
        if not (p := Path(path)).exists():
            raise ValueError(f"missing file {path}")
        table = pq.read_table(p)
        df = table.to_pandas()
        metadata_json = table.schema.metadata["solarpandas".encode()]
        metadata = {}
        for key, value in json.loads(metadata_json).items():
            if key.endswith("/ndarray"):
                metadata[key.split("/", 1)[0]] = np.array(value)
            elif key.endswith("/dataframe"):
                metadata[key.split("/", 1)[0]] = pd.DataFrame(value)
            else:
                metadata[key] = value

        return cls(
            data=df,
            latitude=float(metadata.pop("latitude")),
            longitude=float(metadata.pop("longitude")),
            elevation=float(metadata.pop("elevation")),
            custom_metadata=metadata,
        )
