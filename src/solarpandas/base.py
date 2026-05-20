
import copy
import json
import linecache
from numbers import Number
from typing import Self, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from .types import Elevation, Latitude, Longitude, validate_type

logger.enable(__name__)


def read_csv(filename, **kwargs):
    return SolarDataFrame.read_csv(filename, **kwargs)


def read_parquet(filename):
    return SolarDataFrame.read_parquet(filename)


class SolarSeries(pd.Series):
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
        elevation: Elevation = 0.,
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

    def clone(self, other: pd.Series | pd.DataFrame | Sequence[Number] | Number) -> Self:
        return self.__class__(
            data=np.full((len(self),), other) if isinstance(other, Number) else copy.copy(other),
            latitude=self.latitude,
            longitude=self.longitude,
            elevation=self.elevation,
            custom_metadata=copy.deepcopy(self.custom_metadata))

    # def iplot(self, *args, time_ref: str = "lst", **kwargs):
    #     from .viz_helpers import on_key_pressed_daily_step, onscroll_daily_step

    #     if time_ref.casefold() == "lst":
    #         df = self.set_axis(self.index + pd.Timedelta(self.site_lon * 4, "min"))
    #     if time_ref.casefold() in ("tst", "lat"):
    #         df = self.set_axis(self.sp.true_solar_time)
    #     ax = super(SynSeries, df).plot(*args, **(kwargs | {"kind": "line"}))
    #     ax.get_figure().canvas.mpl_connect("scroll_event", onscroll_daily_step)
    #     ax.get_figure().canvas.mpl_connect("key_press_event", on_key_pressed_daily_step)
    #     return ax

    # def plot_diurnal(self, *args, **kwargs):
    #     # from .viz_helpers import MyDateLocator
    #     from .viz_helpers import MyDateFormatter

    #     df = self.drop_nighttime(max_sza=kwargs.pop("max_sza", 90))
    #     name = df.columns.drop("times")[0]
    #     ax = super(SynSeries, df[name]).plot(*args, **(kwargs | {"kind": "line"}))
    #     # ax.xaxis.set_major_locator(MyDateLocator(12, df.times))
    #     ax.xaxis.set_major_formatter(MyDateFormatter(df.times))
    #     return ax

    # def datemap(self, time_ref: str = "lst", **kwargs):
    #     from matplotlib.dates import DateFormatter

    #     column_name = self.name or "unnamed"
    #     max_sza = kwargs.pop("max_sza", get_option("max_sza"))
    #     df = self.to_frame(column_name).where(self.sp.sza < max_sza, float("nan"))

    #     if time_ref.casefold() == "lst":
    #         df = df.assign(time_ref=df.index + pd.Timedelta(df.site_lon * 4, "min"))

    #     df = (
    #         df.assign(date=df.time_ref.dt.date, time=df.time_ref.dt.time)
    #         .drop(columns="time_ref")
    #         .pivot(index="time", columns="date")
    #         .get(column_name)
    #     )

    #     time_to_minutes = lambda t: int(t.hour * 60 + t.minute + t.second / 60)  # noqa: E731
    #     y = df.index.map(lambda t: np.datetime64(time_to_minutes(t), "m"))

    #     ax = kwargs.pop("ax", pl.gca())
    #     artist = ax.pcolormesh(df.columns, y, df, **kwargs)
    #     ax.yaxis.set_major_formatter(DateFormatter("%H:%M"))
    #     ax.set(
    #         ylabel={
    #             "lst": "Local Solar Time",
    #             "tst": "True Solar Time",
    #             "lat": "Local Apparent Time",
    #         }.get(time_ref.casefold(), "Coordinated Universal Time")
    #     )
    #     return artist

    def __repr__(self):
        epilogue = "\n[latitude={0:.4f}\u00b0 longitude={1:.4f}\u00b0 elevation={2:.1f} m]"
        return (super().__repr__().removesuffix(epilogue)
                + epilogue.format(self.latitude, self.longitude, self.elevation))

    def __str__(self):
        epilogue = "\n[latitude={0:.4f}\u00b0 longitude={1:.4f}\u00b0 elevation={2:.1f} m]"
        return (super().__str__().removesuffix(epilogue)
                + epilogue.format(self.latitude, self.longitude, self.elevation))


class SolarDataFrame(pd.DataFrame):
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
        elevation: Elevation = 0.,
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
        return pd.DataFrame(self)

    def describe(self):
        return self.as_pandas().describe()

    def clone(self, other: pd.Series | pd.DataFrame | Sequence[Number] | Number) -> Self:
        return self.__class__(
            data=np.full((len(self),), other) if isinstance(other, Number) else copy.copy(other),
            latitude=self.latitude,
            longitude=self.longitude,
            elevation=self.elevation,
            custom_metadata=copy.deepcopy(self.custom_metadata))

    def __repr__(self):
        epilogue = "\n[latitude={0:.4f}\u00b0 longitude={1:.4f}\u00b0 elevation={2:.1f} m]"
        return (super().__repr__().removesuffix(epilogue)
                + epilogue.format(self.latitude, self.longitude, self.elevation))

    def __str__(self):
        epilogue = "\n[latitude={0:.4f}\u00b0 longitude={1:.4f}\u00b0 elevation={2:.1f} m]"
        return (super().__repr__().removesuffix(epilogue)
                + epilogue.format(self.latitude, self.longitude, self.elevation))

    def to_csv(self, path: str | Path, **kwargs):
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
        metadata = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "elevation": self.elevation
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
            custom_metadata=metadata
        )
