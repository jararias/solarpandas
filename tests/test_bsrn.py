
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import solarpandas as sp
from solarpandas.origin.bsrn.core import (
    _availability_to_year_table,
    _decode_bsrn_two_digit_year,
    get_database_path,
)
from solarpandas.origin.bsrn.types import (
    DataLogicalRecordName,
    LogicalRecordName,
    Month,
    Site,
    Year,
    validate_type,
)


# ---------------------------------------------------------------------------
# Type validators
# ---------------------------------------------------------------------------


class TestSiteValidator:
    def test_valid_site(self):
        assert validate_type("car", Site) == "car"

    def test_valid_site_bon(self):
        assert validate_type("bon", Site) == "bon"

    def test_invalid_site_uppercase(self):
        with pytest.raises(ValueError):
            validate_type("CAR", Site)

    def test_invalid_site_too_short(self):
        with pytest.raises(ValueError):
            validate_type("ca", Site)

    def test_invalid_site_too_long(self):
        with pytest.raises(ValueError):
            validate_type("cars", Site)

    def test_invalid_site_digits(self):
        with pytest.raises(ValueError):
            validate_type("ca1", Site)

    def test_invalid_site_not_string(self):
        with pytest.raises(TypeError):
            validate_type(123, Site)


class TestYearValidator:
    def test_valid_year(self):
        assert validate_type(2016, Year) == 2016

    def test_valid_year_boundary_min(self):
        assert validate_type(1980, Year) == 1980

    def test_valid_year_boundary_max(self):
        assert validate_type(2100, Year) == 2100

    def test_invalid_year_too_small(self):
        with pytest.raises(ValueError):
            validate_type(1979, Year)

    def test_invalid_year_too_large(self):
        with pytest.raises(ValueError):
            validate_type(2101, Year)

    def test_invalid_year_not_number(self):
        with pytest.raises(TypeError):
            validate_type("abc", Year)


class TestMonthValidator:
    def test_valid_month_all(self):
        for m in range(1, 13):
            assert validate_type(m, Month) == m

    def test_invalid_month_zero(self):
        with pytest.raises(ValueError):
            validate_type(0, Month)

    def test_invalid_month_thirteen(self):
        with pytest.raises(ValueError):
            validate_type(13, Month)


class TestLogicalRecordValidator:
    def test_valid_lr(self):
        assert validate_type("LR0100", LogicalRecordName) == "LR0100"

    def test_valid_lr_uppercase_conversion(self):
        with pytest.raises(ValueError):
            validate_type("lr0100", LogicalRecordName)

    def test_invalid_lr_bad_format(self):
        with pytest.raises(ValueError):
            validate_type("LRX100", LogicalRecordName)

    def test_invalid_lr_too_short(self):
        with pytest.raises(ValueError):
            validate_type("LR010", LogicalRecordName)


class TestDataLogicalRecordValidator:
    def test_valid_lr0100(self):
        assert validate_type("LR0100", DataLogicalRecordName) == "LR0100"

    def test_valid_lr0300(self):
        assert validate_type("LR0300", DataLogicalRecordName) == "LR0300"

    def test_valid_lr0500(self):
        assert validate_type("LR0500", DataLogicalRecordName) == "LR0500"

    def test_invalid_lr0200(self):
        with pytest.raises(ValueError):
            validate_type("LR0200", DataLogicalRecordName)

    def test_invalid_lr0400(self):
        with pytest.raises(ValueError):
            validate_type("LR0400", DataLogicalRecordName)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestDecodeTwoDigitYear:
    def test_current_century_small_year(self):
        result = _decode_bsrn_two_digit_year(16, current_year=2026)
        assert result == 2016

    def test_current_century_pivot(self):
        # pivot = (2026 % 100) + 1 = 27; years <= 27 -> 2000s
        result = _decode_bsrn_two_digit_year(27, current_year=2026)
        assert result == 2027

    def test_previous_century_large_year(self):
        # years > 27 -> 1900s
        result = _decode_bsrn_two_digit_year(92, current_year=2026)
        assert result == 1992

    def test_year_zero(self):
        result = _decode_bsrn_two_digit_year(0, current_year=2026)
        assert result == 2000


class TestAvailabilityToYearTable:
    def _make_availability(self):
        return {
            "car": [
                "car/car0116.dat.gz",
                "car/car0216.dat.gz",
                "car/car0692.dat.gz",
            ],
            "bon": [
                "bon/bon0116.dat.gz",
            ],
        }

    def test_returns_string(self):
        table = _availability_to_year_table(self._make_availability())
        assert isinstance(table, str)

    def test_empty_dict_returns_message(self):
        table = _availability_to_year_table({})
        assert "No BSRN data" in table

    def test_contains_site_names(self):
        table = _availability_to_year_table(self._make_availability())
        assert "car" in table
        assert "bon" in table

    def test_invalid_fill_char_raises(self):
        with pytest.raises(ValueError):
            _availability_to_year_table(self._make_availability(), fill_char="##")

    def test_custom_fill_char(self):
        table = _availability_to_year_table(self._make_availability(), fill_char="*")
        assert "*" in table


# ---------------------------------------------------------------------------
# get_database_path
# ---------------------------------------------------------------------------


class TestGetDatabasePath:
    def test_returns_path(self):
        path = get_database_path()
        assert isinstance(path, Path)

    def test_path_ends_with_bsrn(self):
        path = get_database_path()
        assert path.name == "bsrn"


# ---------------------------------------------------------------------------
# data_availability with a mocked cache file
# ---------------------------------------------------------------------------


class TestDataAvailabilityWithCache:
    def test_reads_from_existing_cache(self, tmp_path, monkeypatch):
        from solarpandas.origin.bsrn import core

        availability = {"car": ["car/car0116.dat.gz"], "bon": ["bon/bon0116.dat.gz"]}
        cache_dir = tmp_path / "ftp"
        cache_dir.mkdir(parents=True)
        cache_file = cache_dir / "availability.json"
        cache_file.write_text(json.dumps(availability))

        monkeypatch.setattr(core, "get_database_path", lambda: tmp_path)

        result = core.data_availability(update=False)
        assert isinstance(result, dict)
        assert "car" in result
        assert "bon" in result

    def test_returns_year_table_when_requested(self, tmp_path, monkeypatch):
        from solarpandas.origin.bsrn import core

        availability = {"car": ["car/car0116.dat.gz"]}
        cache_dir = tmp_path / "ftp"
        cache_dir.mkdir(parents=True)
        (cache_dir / "availability.json").write_text(json.dumps(availability))

        monkeypatch.setattr(core, "get_database_path", lambda: tmp_path)

        result = core.data_availability(update=False, as_year_table=True)
        assert isinstance(result, str)
        assert "car" in result


# ---------------------------------------------------------------------------
# Sample data (no network required)
# ---------------------------------------------------------------------------


class TestSampleData:
    def test_load_carpentras_returns_solar_dataframe(self, carpentras_data):
        assert isinstance(carpentras_data, sp.SolarDataFrame)

    def test_carpentras_has_location(self, carpentras_data):
        assert -90 < carpentras_data.latitude < 90
        assert -180 <= carpentras_data.longitude < 180

    def test_carpentras_has_radiation_columns(self, carpentras_data):
        # Sample data has at minimum ghi and dni from LR0100
        expected = {"ghi", "dni"}
        assert expected.issubset(set(carpentras_data.columns))

    def test_carpentras_has_datetime_index(self, carpentras_data):
        assert isinstance(carpentras_data.index, pd.DatetimeIndex)

    def test_carpentras_has_1min_frequency(self, carpentras_data):
        from solarpandas.helpers import infer_time_step
        ts = infer_time_step(carpentras_data)
        assert ts == pd.Timedelta("1min")

    def test_carpentras_latitude_approx(self, carpentras_data):
        # Carpentras is at ~44°N
        assert 43.0 < carpentras_data.latitude < 45.0

    def test_carpentras_has_custom_metadata(self, carpentras_data):
        meta = carpentras_data.custom_metadata
        assert isinstance(meta, dict)
        assert len(meta) > 0


# ---------------------------------------------------------------------------
# Network-dependent tests (skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.network
class TestBSRNNetworkAccess:
    def test_data_availability_returns_dict(self):
        from solarpandas.origin.bsrn import data_availability
        result = data_availability(update=True)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_load_metadata_returns_dict(self):
        from solarpandas.origin.bsrn import load_metadata
        metadata = load_metadata(update=True)
        assert isinstance(metadata, dict)
        assert len(metadata) > 0

    def test_load_data_returns_solar_dataframe(self):
        from solarpandas.origin.bsrn import load_data
        data = load_data("car", years=2016, logical_record="LR0100")
        assert isinstance(data, sp.SolarDataFrame)
        assert len(data) > 0
