import pandas as pd
import pytest

import solarpandas as sp

EXPECTED_TESTS = [
    "ghi_ppl",
    "dif_ppl",
    "dni_ppl",
    "ghi_erl",
    "dif_erl",
    "dni_erl",
    "Kn_ppl",
    "Kn_erl",
    "KT_erl",
    "K_erl",
    "K_erl_clear",
    "closure",
    "trackeroff",
]


class TestQualityControlAccessorBasics:
    def test_requires_solar_dataframe(self):
        plain = pd.DataFrame({"ghi": [100.0, 200.0]})
        with pytest.raises(AttributeError):
            plain.qc.tests  # noqa: B018

    def test_tests_property_is_dataframe(self, carpentras_data):
        tests = carpentras_data.qc.tests
        assert isinstance(tests, pd.DataFrame)

    def test_all_expected_tests_present(self, carpentras_data):
        columns = carpentras_data.qc.tests.columns.tolist()
        for name in EXPECTED_TESTS:
            assert name in columns, f"Expected test '{name}' not found in QC results"

    def test_tests_same_length_as_data(self, carpentras_data):
        assert len(carpentras_data.qc.tests) == len(carpentras_data)

    def test_getitem_returns_series(self, carpentras_data):
        result = carpentras_data.qc.tests["ghi_ppl"]
        assert isinstance(result, pd.Series)

    def test_getattr_returns_series(self, carpentras_data):
        result = carpentras_data.qc.ghi_ppl
        assert isinstance(result, pd.Series)

    def test_getitem_unknown_test_raises(self, carpentras_data):
        with pytest.raises(KeyError):
            carpentras_data.qc.tests["nonexistent_test"]

    def test_getattr_unknown_test_raises(self, carpentras_data):
        with pytest.raises(AttributeError):
            carpentras_data.qc.nonexistent_test  # noqa: B018


class TestQCValues:
    def test_test_values_are_valid_flags(self, carpentras_data):
        tests = carpentras_data.qc.tests
        valid_values = {-1, 0, 1}
        for col in tests.columns:
            unique_vals = set(tests[col].unique())
            assert unique_vals <= valid_values, (
                f"Column '{col}' contains unexpected values: {unique_vals - valid_values}"
            )

    def test_ghi_ppl_qcflag_dtype(self, carpentras_data):
        from solarpandas.types import QCFlagDtype

        result = carpentras_data.qc.ghi_ppl
        assert isinstance(result.dtype, QCFlagDtype)


class TestQCFilter:
    def test_filter_by_ghi_component(self, carpentras_data):
        filtered = carpentras_data.qc.filter("ghi")
        assert isinstance(filtered, pd.DataFrame)
        assert len(filtered.columns) > 0
        for col in filtered.columns:
            assert "ghi" in col or col in (
                "Kn_ppl",
                "Kn_erl",
                "KT_erl",
                "K_erl",
                "K_erl_clear",
                "trackeroff",
                "closure",
            )

    def test_filter_by_dni_component(self, carpentras_data):
        filtered = carpentras_data.qc.filter("dni")
        assert len(filtered.columns) > 0

    def test_filter_by_dif_component(self, carpentras_data):
        filtered = carpentras_data.qc.filter("dif")
        assert len(filtered.columns) > 0

    def test_filter_by_like(self, carpentras_data):
        filtered = carpentras_data.qc.filter(like="ppl")
        assert all("ppl" in col for col in filtered.columns)

    def test_filter_by_regex(self, carpentras_data):
        filtered = carpentras_data.qc.filter(regex=r"^ghi_")
        assert all(col.startswith("ghi_") for col in filtered.columns)

    def test_filter_by_specific_tests(self, carpentras_data):
        filtered = carpentras_data.qc.filter(tests=["ghi_ppl", "dni_ppl"])
        assert list(filtered.columns) == ["ghi_ppl", "dni_ppl"]

    def test_filter_invalid_component_raises(self, carpentras_data):
        with pytest.raises(ValueError):
            carpentras_data.qc.filter("invalid_component")


class TestQCFailedPassed:
    def test_failed_returns_bool_series(self, carpentras_data):
        failed = carpentras_data.qc.failed()
        assert isinstance(failed, pd.Series)
        assert failed.dtype == bool

    def test_passed_returns_bool_series(self, carpentras_data):
        passed = carpentras_data.qc.passed()
        assert isinstance(passed, pd.Series)
        assert passed.dtype == bool

    def test_failed_same_length_as_data(self, carpentras_data):
        assert len(carpentras_data.qc.failed()) == len(carpentras_data)

    def test_passed_and_failed_are_complementary(self, carpentras_data):
        # A row cannot simultaneously pass and fail all tests
        failed = carpentras_data.qc.failed()
        passed = carpentras_data.qc.passed()
        # Where data failed, it cannot have passed all tests
        assert not (failed & passed).any(), "Some rows are both failed and passed"

    def test_failed_by_component(self, carpentras_data):
        failed_ghi = carpentras_data.qc.failed("ghi")
        assert isinstance(failed_ghi, pd.Series)
        assert failed_ghi.dtype == bool

    def test_passed_by_component(self, carpentras_data):
        passed_ghi = carpentras_data.qc.passed("ghi")
        assert isinstance(passed_ghi, pd.Series)
        assert passed_ghi.dtype == bool


class TestQCMaskFailed:
    def test_mask_failed_returns_dataframe(self, carpentras_data):
        masked = carpentras_data.qc.mask_failed()
        assert isinstance(masked, pd.DataFrame)

    def test_mask_failed_same_shape(self, carpentras_data):
        masked = carpentras_data.qc.mask_failed()
        assert masked.shape == carpentras_data.shape

    def test_mask_failed_by_component(self, carpentras_data):
        if "ghi" in carpentras_data.columns:
            masked = carpentras_data.qc.mask_failed("ghi")
            failed = carpentras_data.qc.failed("ghi")
            if failed.any():
                assert masked["ghi"][failed].isna().all()


class TestQCCache:
    def test_clear_cache(self):
        sp.clear_qc_cache()
        info = sp.get_qc_cache_info()
        assert info["current_size"] == 0

    def test_cache_info_keys(self):
        info = sp.get_qc_cache_info()
        assert set(info.keys()) == {"hits", "misses", "current_size", "max_size"}

    def test_cache_hit_on_repeated_access(self, carpentras_data):
        sp.clear_qc_cache()
        _ = carpentras_data.qc.tests
        info_first = sp.get_qc_cache_info()
        _ = carpentras_data.qc.failed()
        info_second = sp.get_qc_cache_info()
        assert info_second["hits"] > info_first["hits"]


class TestQCFlagAccessor:
    """Tests for the .flag accessor on Series with QCFlagDtype."""

    @pytest.fixture
    def flag_series(self, carpentras_data):
        return carpentras_data.qc.ghi_ppl

    def test_accessor_unavailable_on_plain_series(self, carpentras_data):
        with pytest.raises(TypeError):
            carpentras_data["ghi"].flag  # noqa: B018

    def test_fails_is_boolean_series(self, flag_series):
        assert flag_series.flag.fails.dtype == bool

    def test_passes_is_boolean_series(self, flag_series):
        assert flag_series.flag.passes.dtype == bool

    def test_not_verifiable_is_boolean_series(self, flag_series):
        assert flag_series.flag.not_verifiable.dtype == bool

    def test_fails_length_matches_source(self, flag_series):
        assert len(flag_series.flag.fails) == len(flag_series)

    def test_passes_length_matches_source(self, flag_series):
        assert len(flag_series.flag.passes) == len(flag_series)

    def test_not_verifiable_length_matches_source(self, flag_series):
        assert len(flag_series.flag.not_verifiable) == len(flag_series)

    def test_categories_are_mutually_exclusive(self, flag_series):
        fails = flag_series.flag.fails
        passes = flag_series.flag.passes
        not_verifiable = flag_series.flag.not_verifiable
        assert not (fails & passes).any(), "Some rows are both failed and passed"
        assert not (fails & not_verifiable).any(), (
            "Some rows are both failed and not_verifiable"
        )
        assert not (passes & not_verifiable).any(), (
            "Some rows are both passed and not_verifiable"
        )

    def test_categories_are_exhaustive(self, flag_series):
        fails = flag_series.flag.fails
        passes = flag_series.flag.passes
        not_verifiable = flag_series.flag.not_verifiable
        assert (fails | passes | not_verifiable).all(), (
            "Some rows fall into no category"
        )

    def test_counts_returns_series(self, flag_series):
        counts = flag_series.flag.counts(skip_nighttime=False)
        assert isinstance(counts, pd.Series)

    def test_counts_sum_equals_length(self, flag_series):
        counts = flag_series.flag.counts(skip_nighttime=False)
        assert counts.sum() == len(flag_series)

    def test_counts_skip_nighttime_default(self, flag_series):
        counts_day = flag_series.flag.counts(skip_nighttime=True)
        counts_all = flag_series.flag.counts(skip_nighttime=False)
        assert counts_day.sum() <= counts_all.sum()

    def test_counts_index_contains_flag_names(self, flag_series):
        counts = flag_series.flag.counts(skip_nighttime=False)
        valid = {"FAILED", "PASSED", "NOT_VERIFIABLE"}
        assert set(counts.index).issubset(valid)
