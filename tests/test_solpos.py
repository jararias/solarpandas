
import numpy as np
import pandas as pd
import pytest

import solarpandas as sp


class TestSolarPositionProperties:
    def test_sza_is_solar_series(self, solar_series):
        result = solar_series.solpos.sza
        assert isinstance(result, sp.SolarSeries)

    def test_sza_has_same_index(self, solar_series):
        result = solar_series.solpos.sza
        assert result.index.equals(solar_series.index)

    def test_zenith_equals_sza(self, solar_series):
        sza = solar_series.solpos.sza
        zenith = solar_series.solpos.zenith
        assert sza.equals(zenith)

    def test_elevation_is_90_minus_zenith(self, solar_series):
        zenith = solar_series.solpos.zenith
        elev = solar_series.solpos.elevation
        np.testing.assert_allclose(elev.values, 90.0 - zenith.values, atol=1e-10)

    def test_sza_daytime_range(self, solar_series):
        sza = solar_series.solpos.sza
        # All SZA values must be in [0, 180] range
        assert (sza >= 0).all()
        assert (sza <= 180).all()

    def test_azimuth_range(self, solar_series):
        az = solar_series.solpos.azimuth
        # sunwhere returns azimuth in [-180, 180] convention
        assert (az >= -180).all()
        assert (az <= 180).all()

    def test_cosz_range(self, solar_series):
        cosz = solar_series.solpos.cosz
        assert (cosz >= -1.0).all()
        assert (cosz <= 1.0).all()

    def test_eth_nonneg(self, solar_series):
        eth = solar_series.solpos.eth
        # ETH >= 0 always (0 at night when cosz <= 0)
        assert (eth >= 0).all()

    def test_etn_approx_isc_times_ecf(self, solar_series):
        ISC = 1361.1
        etn = solar_series.solpos.etn
        ecf = solar_series.solpos.ecf
        np.testing.assert_allclose(etn.values, ISC * ecf.values, rtol=1e-6)

    def test_ecf_near_unity(self, solar_series):
        ecf = solar_series.solpos.ecf
        # Earth-Sun distance correction factor varies by ±3.4% around 1.0
        assert (ecf > 0.96).all()
        assert (ecf < 1.05).all()

    def test_true_solar_time_is_datetime(self, solar_series):
        tst = solar_series.solpos.tst
        assert isinstance(tst, sp.SolarSeries)
        assert pd.api.types.is_datetime64_any_dtype(tst)

    def test_tst_alias(self, solar_series):
        assert solar_series.solpos.tst.equals(solar_series.solpos.true_solar_time)

    def test_local_solar_time(self, solar_series):
        lst = solar_series.solpos.lst
        lon = solar_series.longitude
        expected_shift = pd.Timedelta(lon * 4, "min")
        utc_times = solar_series.index
        np.testing.assert_array_equal(
            lst.values,
            (utc_times + expected_shift).values,
        )

    def test_tsd_is_tst_floored_to_day(self, solar_series):
        tst = solar_series.solpos.tst
        tsd = solar_series.solpos.tsd
        floored = tst.dt.floor("D")
        assert tsd.equals(floored)

    def test_solpos_on_dataframe(self, solar_dataframe):
        sza = solar_dataframe.solpos.sza
        assert isinstance(sza, sp.SolarSeries)
        assert len(sza) == len(solar_dataframe)

    def test_solpos_metadata_propagation(self, solar_series):
        sza = solar_series.solpos.sza
        assert sza.latitude == solar_series.latitude
        assert sza.longitude == solar_series.longitude
        assert sza.elevation == solar_series.elevation

    def test_invalid_object_raises(self):
        plain = pd.Series([1, 2, 3])
        with pytest.raises(AttributeError):
            plain.solpos.sza  # noqa: B018


class TestSolarPositionCompute:
    def test_compute_returns_sunpos(self, solar_series):
        import sunwhere
        result = solar_series.solpos.compute()
        assert isinstance(result, sunwhere._base.Sunpos)

    def test_compute_psa_algorithm(self, solar_series):
        result = solar_series.solpos.compute(algorithm="psa")
        assert result is not None

    def test_compute_no_refraction(self, solar_series):
        result = solar_series.solpos.compute(refraction=False)
        assert result is not None


class TestSolarPositionSunriseSunset:
    def test_sunrise_utc(self, solar_series):
        sr = solar_series.solpos.sunrise()
        assert pd.api.types.is_datetime64_any_dtype(sr)
        assert len(sr) == len(solar_series)

    def test_sunset_utc(self, solar_series):
        ss = solar_series.solpos.sunset()
        assert pd.api.types.is_datetime64_any_dtype(ss)
        assert len(ss) == len(solar_series)

    def test_sunrise_before_sunset(self, solar_series):
        sr = solar_series.solpos.sunrise()
        ss = solar_series.solpos.sunset()
        # Drop NaN values (nighttime points have no sunrise/sunset)
        mask = sr.notna() & ss.notna()
        assert (sr[mask] < ss[mask]).all()


class TestSolarPositionCache:
    def test_clear_cache(self):
        sp.clear_solpos_cache()
        info = sp.get_solpos_cache_info()
        assert info["current_size"] == 0

    def test_cache_info_keys(self):
        info = sp.get_solpos_cache_info()
        assert set(info.keys()) == {"hits", "misses", "current_size", "max_size"}

    def test_cache_hit_on_second_access(self, solar_series):
        sp.clear_solpos_cache()
        _ = solar_series.solpos.sza
        info_after_first = sp.get_solpos_cache_info()
        _ = solar_series.solpos.zenith
        info_after_second = sp.get_solpos_cache_info()
        assert info_after_second["hits"] > info_after_first["hits"]
