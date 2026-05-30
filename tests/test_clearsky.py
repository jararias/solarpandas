
import numpy as np
import pandas as pd
import pytest

import solarpandas as sp


@pytest.fixture(autouse=True)
def clear_clearsky_cache_before():
    sp.clear_clearsky_cache()
    yield
    sp.clear_clearsky_cache()


class TestClearskyPropertiesViaLTA:
    """Test clearsky properties using the LTA accessor (no credentials required)."""

    def test_ghi_is_solar_series(self, solar_series):
        result = solar_series.lta.ghi
        assert isinstance(result, sp.SolarSeries)

    def test_ghi_same_index(self, solar_series):
        result = solar_series.lta.ghi
        assert result.index.equals(solar_series.index)

    def test_ghi_nonneg(self, solar_series):
        ghi = solar_series.lta.ghi
        assert (ghi >= -0.1).all()  # allow small numerical errors near sunrise/sunset

    def test_dni_nonneg(self, solar_series):
        dni = solar_series.lta.dni
        assert (dni >= -0.1).all()

    def test_dif_nonneg(self, solar_series):
        dif = solar_series.lta.dif
        assert (dif >= -0.1).all()

    def test_ghi_bounded_above(self, solar_series):
        ghi = solar_series.lta.ghi
        assert (ghi <= 1400).all()

    def test_dni_bounded_above(self, solar_series):
        dni = solar_series.lta.dni
        assert (dni <= 1400).all()

    def test_csi_is_solar_series(self, solar_series):
        csi = solar_series.lta.csi
        assert isinstance(csi, sp.SolarSeries)

    def test_metadata_preserved(self, solar_series):
        ghi = solar_series.lta.ghi
        assert ghi.latitude == solar_series.latitude
        assert ghi.longitude == solar_series.longitude

    def test_lta_on_dataframe(self, solar_dataframe):
        ghi = solar_dataframe.lta.ghi
        assert isinstance(ghi, sp.SolarSeries)
        assert len(ghi) == len(solar_dataframe)


class TestClearskyAccessorWithPatchedConfig:
    """Test the general .clearsky accessor using merra2_lta atmosphere via config patch."""

    @pytest.fixture(autouse=True)
    def patch_clearsky_atmosphere(self, monkeypatch):
        from solarpandas import config as sp_config
        original = sp_config.get_option("clearsky.atmosphere", default="crs_soda")
        sp_config.set_option("clearsky.atmosphere", "merra2_lta")
        yield
        sp_config.set_option("clearsky.atmosphere", original)

    def test_clearsky_ghi_nonneg(self, solar_series):
        ghi = solar_series.clearsky.ghi
        assert (ghi >= -0.1).all()

    def test_clearsky_returns_solar_series(self, solar_series):
        ghi = solar_series.clearsky.ghi
        assert isinstance(ghi, sp.SolarSeries)

    def test_clearsky_invalid_object_raises(self):
        plain = pd.Series([1, 2, 3])
        with pytest.raises(AttributeError):
            plain.clearsky.ghi  # noqa: B018


class TestCDAAccessor:
    def test_cda_ghi_nonneg(self, solar_series):
        ghi = solar_series.cda.ghi
        assert isinstance(ghi, sp.SolarSeries)
        assert (ghi >= -1.0).all()  # small numerical errors near sunrise/sunset

    def test_cda_dni_nonneg(self, solar_series):
        dni = solar_series.cda.dni
        assert (dni >= -1.0).all()


class TestClearskyCache:
    def test_clear_cache(self):
        info = sp.get_clearsky_cache_info()
        assert info["current_size"] == 0

    def test_cache_info_keys(self):
        info = sp.get_clearsky_cache_info()
        assert set(info.keys()) == {"hits", "misses", "current_size", "max_size"}

    def test_cache_hit_on_second_access(self, solar_series):
        _ = solar_series.lta.ghi
        info_after_first = sp.get_clearsky_cache_info()
        _ = solar_series.lta.dni
        info_after_second = sp.get_clearsky_cache_info()
        assert info_after_second["hits"] > info_after_first["hits"]
