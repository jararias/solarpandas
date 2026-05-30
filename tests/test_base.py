
import copy

import numpy as np
import pandas as pd
import pytest

import solarpandas as sp


LOC = {"latitude": 44.083, "longitude": 5.059, "elevation": 100.0}
IDX = pd.date_range("2016-06-21", periods=10, freq="1min", tz="UTC")
DATA = np.arange(10, dtype=float)


# ---------------------------------------------------------------------------
# SolarSeries
# ---------------------------------------------------------------------------


class TestSolarSeriesConstruction:
    def test_is_pandas_series(self):
        s = sp.SolarSeries(DATA, index=IDX, **LOC)
        assert isinstance(s, pd.Series)

    def test_is_solar_series(self):
        s = sp.SolarSeries(DATA, index=IDX, **LOC)
        assert isinstance(s, sp.SolarSeries)

    def test_coordinates_stored(self):
        s = sp.SolarSeries(DATA, index=IDX, **LOC)
        assert s.latitude == LOC["latitude"]
        assert s.longitude == LOC["longitude"]
        assert s.elevation == LOC["elevation"]

    def test_default_elevation(self):
        s = sp.SolarSeries(DATA, index=IDX, latitude=44.0, longitude=5.0)
        assert s.elevation == 0.0

    def test_invalid_latitude_too_high(self):
        with pytest.raises((ValueError, TypeError)):
            sp.SolarSeries(DATA, index=IDX, latitude=91.0, longitude=0.0)

    def test_invalid_latitude_too_low(self):
        with pytest.raises((ValueError, TypeError)):
            sp.SolarSeries(DATA, index=IDX, latitude=-90.0, longitude=0.0)

    def test_invalid_longitude_too_high(self):
        with pytest.raises((ValueError, TypeError)):
            sp.SolarSeries(DATA, index=IDX, latitude=0.0, longitude=181.0)

    def test_invalid_elevation_too_high(self):
        with pytest.raises((ValueError, TypeError)):
            sp.SolarSeries(DATA, index=IDX, latitude=0.0, longitude=0.0, elevation=9000.0)

    def test_reserved_key_latitude_in_metadata(self):
        with pytest.raises(ValueError, match="latitude"):
            sp.SolarSeries(DATA, index=IDX, **LOC, custom_metadata={"latitude": 1.0})

    def test_reserved_key_longitude_in_metadata(self):
        with pytest.raises(ValueError, match="longit"):
            sp.SolarSeries(DATA, index=IDX, **LOC, custom_metadata={"longitude": 1.0})

    def test_reserved_key_elevation_in_metadata(self):
        with pytest.raises(ValueError, match="elevation"):
            sp.SolarSeries(DATA, index=IDX, **LOC, custom_metadata={"elevation": 1.0})

    def test_custom_metadata_stored(self):
        meta = {"source": "test"}
        s = sp.SolarSeries(DATA, index=IDX, **LOC, custom_metadata=meta)
        assert s.custom_metadata["source"] == "test"


class TestSolarSeriesMetadataPropagation:
    def _make(self, data=DATA):
        return sp.SolarSeries(data, index=IDX, **LOC)

    def test_slicing_preserves_metadata(self):
        s = self._make()
        sliced = s.iloc[:5]
        assert isinstance(sliced, sp.SolarSeries)
        assert sliced.latitude == LOC["latitude"]
        assert sliced.longitude == LOC["longitude"]
        assert sliced.elevation == LOC["elevation"]

    def test_arithmetic_preserves_metadata(self):
        s = self._make()
        result = s * 2.0
        assert isinstance(result, sp.SolarSeries)
        assert result.latitude == LOC["latitude"]

    def test_dropna_preserves_type(self):
        data = DATA.copy().astype(float)
        data[3] = np.nan
        s = sp.SolarSeries(data, index=IDX, **LOC)
        dropped = s.dropna()
        assert isinstance(dropped, sp.SolarSeries)

    def test_expand_to_dataframe_returns_solar_dataframe(self):
        s = self._make()
        df = s.to_frame()
        assert isinstance(df, sp.SolarDataFrame)
        assert df.latitude == LOC["latitude"]


class TestSolarSeriesClone:
    def _make(self, meta=None):
        return sp.SolarSeries(DATA, index=IDX, **LOC, custom_metadata=meta or {})

    def test_clone_with_scalar(self):
        s = self._make()
        c = s.clone(5.0)
        assert isinstance(c, sp.SolarSeries)
        assert (c == 5.0).all()
        assert c.latitude == LOC["latitude"]

    def test_clone_with_array(self):
        s = self._make()
        arr = np.ones(len(s)) * 3.0
        c = s.clone(arr)
        assert (c == 3.0).all()

    def test_clone_with_series(self):
        s = self._make()
        other = pd.Series(np.ones(len(s)) * 7.0, index=IDX)
        c = s.clone(other)
        assert (c == 7.0).all()

    def test_clone_deep_copies_metadata(self):
        meta = {"source": "test"}
        s = self._make(meta=meta)
        c = s.clone(s)
        c.custom_metadata["source"] = "modified"
        assert s.custom_metadata["source"] == "test"


class TestSolarSeriesRepr:
    def test_repr_contains_coordinates(self):
        s = sp.SolarSeries(DATA, index=IDX, **LOC)
        r = repr(s)
        assert "44.0830" in r
        assert "5.0590" in r
        assert "100.0" in r


class TestSolarSeriesSerialization:
    def test_csv_round_trip(self, tmp_path):
        s = sp.SolarDataFrame({"value": DATA}, index=IDX, **LOC)
        path = tmp_path / "test.csv"
        s.to_csv(path)
        loaded = sp.SolarDataFrame.read_csv(path)
        assert isinstance(loaded, sp.SolarDataFrame)
        assert loaded.latitude == pytest.approx(LOC["latitude"])
        assert loaded.longitude == pytest.approx(LOC["longitude"])
        assert loaded.elevation == pytest.approx(LOC["elevation"])

    def test_parquet_round_trip(self, tmp_path):
        s = sp.SolarDataFrame({"value": DATA}, index=IDX, **LOC)
        path = tmp_path / "test.parquet"
        s.to_parquet(path)
        loaded = sp.SolarDataFrame.read_parquet(path)
        assert isinstance(loaded, sp.SolarDataFrame)
        assert loaded.latitude == pytest.approx(LOC["latitude"])
        assert loaded.longitude == pytest.approx(LOC["longitude"])
        assert loaded.elevation == pytest.approx(LOC["elevation"])

    def test_parquet_preserves_custom_metadata(self, tmp_path):
        meta = {"network": "BSRN", "station": "CAR"}
        s = sp.SolarDataFrame({"value": DATA}, index=IDX, **LOC, custom_metadata=meta)
        path = tmp_path / "test.parquet"
        s.to_parquet(path)
        loaded = sp.SolarDataFrame.read_parquet(path)
        assert loaded.custom_metadata["network"] == "BSRN"
        assert loaded.custom_metadata["station"] == "CAR"

    def test_read_csv_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="missing file"):
            sp.SolarDataFrame.read_csv(tmp_path / "nonexistent.csv")

    def test_read_parquet_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="missing file"):
            sp.SolarDataFrame.read_parquet(tmp_path / "nonexistent.parquet")


# ---------------------------------------------------------------------------
# SolarDataFrame
# ---------------------------------------------------------------------------


class TestSolarDataFrameConstruction:
    def test_is_pandas_dataframe(self):
        df = sp.SolarDataFrame({"a": DATA}, index=IDX, **LOC)
        assert isinstance(df, pd.DataFrame)

    def test_is_solar_dataframe(self):
        df = sp.SolarDataFrame({"a": DATA}, index=IDX, **LOC)
        assert isinstance(df, sp.SolarDataFrame)

    def test_coordinates_stored(self):
        df = sp.SolarDataFrame({"a": DATA}, index=IDX, **LOC)
        assert df.latitude == LOC["latitude"]
        assert df.longitude == LOC["longitude"]
        assert df.elevation == LOC["elevation"]

    def test_column_access_returns_solar_series(self):
        df = sp.SolarDataFrame({"a": DATA, "b": DATA * 2}, index=IDX, **LOC)
        col = df["a"]
        assert isinstance(col, sp.SolarSeries)
        assert col.latitude == LOC["latitude"]

    def test_row_slice_returns_solar_dataframe(self):
        df = sp.SolarDataFrame({"a": DATA}, index=IDX, **LOC)
        sliced = df.iloc[:5]
        assert isinstance(sliced, sp.SolarDataFrame)
        assert sliced.latitude == LOC["latitude"]

    def test_as_pandas_returns_plain_dataframe(self):
        df = sp.SolarDataFrame({"a": DATA}, index=IDX, **LOC)
        plain = df.as_pandas()
        assert type(plain) is pd.DataFrame
        assert not isinstance(plain, sp.SolarDataFrame)

    def test_describe_returns_plain_dataframe(self):
        df = sp.SolarDataFrame({"a": DATA}, index=IDX, **LOC)
        desc = df.describe()
        assert type(desc) is pd.DataFrame

    def test_invalid_coordinates_raise(self):
        with pytest.raises((ValueError, TypeError)):
            sp.SolarDataFrame({"a": DATA}, index=IDX, latitude=200.0, longitude=0.0)
