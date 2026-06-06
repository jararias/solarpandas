"""Tests for the SolarPlotAccessor (.solarplot accessor)."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to free memory."""
    yield
    plt.close("all")


class TestDiurnalPlot:
    def test_returns_figure_from_series(self, solar_series):
        fig = solar_series.solarplot.diurnal()
        assert isinstance(fig, plt.Figure)

    def test_returns_figure_from_dataframe(self, solar_dataframe):
        fig = solar_dataframe.solarplot.diurnal(column="ghi")
        assert isinstance(fig, plt.Figure)

    def test_multicolumn_list(self, solar_dataframe):
        fig = solar_dataframe.solarplot.diurnal(column=["ghi", "dni"])
        assert isinstance(fig, plt.Figure)

    def test_multicolumn_tuple(self, solar_dataframe):
        fig = solar_dataframe.solarplot.diurnal(column=("ghi", "dif"))
        assert isinstance(fig, plt.Figure)

    def test_no_column_defaults_to_all(self, solar_dataframe):
        fig = solar_dataframe.solarplot.diurnal()
        assert isinstance(fig, plt.Figure)

    def test_custom_max_sza(self, solar_series):
        fig = solar_series.solarplot.diurnal(max_sza=85.0)
        assert isinstance(fig, plt.Figure)

    def test_max_sza_too_restrictive_raises(self, solar_series):
        with pytest.raises(ValueError, match="No daytime samples"):
            solar_series.solarplot.diurnal(max_sza=0.0)

    def test_invalid_plain_series_raises(self):
        plain = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(AttributeError):
            plain.solarplot.diurnal()

    def test_invalid_plain_dataframe_raises(self):
        plain = pd.DataFrame({"ghi": [1.0, 2.0, 3.0]})
        with pytest.raises(AttributeError):
            plain.solarplot.diurnal()


class TestHeatmapPlot:
    def test_returns_figure_from_series(self, solar_series):
        fig = solar_series.solarplot.heatmap()
        assert isinstance(fig, plt.Figure)

    def test_returns_figure_from_dataframe(self, solar_dataframe):
        fig = solar_dataframe.solarplot.heatmap(column="ghi")
        assert isinstance(fig, plt.Figure)

    def test_time_ref_utc(self, solar_series):
        fig = solar_series.solarplot.heatmap(time_ref="utc")
        assert isinstance(fig, plt.Figure)

    def test_time_ref_lst(self, solar_series):
        fig = solar_series.solarplot.heatmap(time_ref="lst")
        assert isinstance(fig, plt.Figure)

    def test_time_ref_tst(self, solar_series):
        fig = solar_series.solarplot.heatmap(time_ref="tst")
        assert isinstance(fig, plt.Figure)

    def test_no_colorbar(self, solar_series):
        fig = solar_series.solarplot.heatmap(colorbar=False)
        assert isinstance(fig, plt.Figure)

    def test_twilight_line(self, solar_series):
        fig = solar_series.solarplot.heatmap(twilight_line=True)
        assert isinstance(fig, plt.Figure)

    def test_no_nighttime_masking(self, solar_series):
        fig = solar_series.solarplot.heatmap(max_sza=None)
        assert isinstance(fig, plt.Figure)

    def test_invalid_plain_series_raises(self):
        plain = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(AttributeError):
            plain.solarplot.heatmap()


class TestRollingPlot:
    """Tests for the .solarplot.rolling() method.

    This method uses ``max_sza`` (not ``margin_min``) to control how far into
    twilight the x-axis extends.
    """

    def test_returns_figure(self, carpentras_data):
        fig = carpentras_data.solarplot.rolling("ghi")
        assert isinstance(fig, plt.Figure)

    def test_no_column_plots_all(self, carpentras_data):
        fig = carpentras_data.solarplot.rolling()
        assert isinstance(fig, plt.Figure)

    def test_single_column_string(self, carpentras_data):
        fig = carpentras_data.solarplot.rolling("dni")
        assert isinstance(fig, plt.Figure)

    def test_multicolumn_list(self, carpentras_data):
        fig = carpentras_data.solarplot.rolling(["ghi", "dni"])
        assert isinstance(fig, plt.Figure)

    def test_multicolumn_tuple(self, carpentras_data):
        fig = carpentras_data.solarplot.rolling(("ghi", "dif"))
        assert isinstance(fig, plt.Figure)

    def test_window_size_1(self, carpentras_data):
        fig = carpentras_data.solarplot.rolling("ghi", window_size=1)
        assert isinstance(fig, plt.Figure)

    def test_window_size_greater_than_available(self, carpentras_data):
        # window_size=3 on a 1-day dataset: clamps to available data, no error
        fig = carpentras_data.solarplot.rolling("ghi", window_size=3)
        assert isinstance(fig, plt.Figure)

    def test_y_scale_per_day(self, carpentras_data):
        fig = carpentras_data.solarplot.rolling("ghi", y_scale="per_day")
        assert isinstance(fig, plt.Figure)

    def test_y_scale_global(self, carpentras_data):
        fig = carpentras_data.solarplot.rolling("ghi", y_scale="global")
        assert isinstance(fig, plt.Figure)

    def test_max_sza_narrow(self, carpentras_data):
        # SZA < 80: midday-only window
        fig = carpentras_data.solarplot.rolling("ghi", max_sza=80.0)
        assert isinstance(fig, plt.Figure)

    def test_max_sza_wide(self, carpentras_data):
        # SZA < 100: includes deep twilight
        fig = carpentras_data.solarplot.rolling("ghi", max_sza=100.0)
        assert isinstance(fig, plt.Figure)

    def test_max_sza_default_95(self, carpentras_data):
        # Explicit default matches implicit default
        fig1 = carpentras_data.solarplot.rolling("ghi")
        fig2 = carpentras_data.solarplot.rolling("ghi", max_sza=95.0)
        # Both should produce figures with a single axes
        assert len(fig1.axes) == len(fig2.axes) == 1

    def test_plot_kwargs_per_column(self, carpentras_data):
        fig = carpentras_data.solarplot.rolling(
            ["ghi", "dni"],
            plot_kwargs={"ghi": {"color": "gold"}, "dni": {"color": "tomato"}},
        )
        assert isinstance(fig, plt.Figure)

    def test_plot_kwargs_global(self, carpentras_data):
        fig = carpentras_data.solarplot.rolling("ghi", lw=2.0, alpha=0.8)
        assert isinstance(fig, plt.Figure)

    def test_step_parameter(self, carpentras_data):
        fig = carpentras_data.solarplot.rolling("ghi", step=2)
        assert isinstance(fig, plt.Figure)

    def test_solar_series_input(self, solar_series):
        fig = solar_series.solarplot.rolling()
        assert isinstance(fig, plt.Figure)

    def test_invalid_window_size_zero_raises(self, carpentras_data):
        with pytest.raises(ValueError, match="window_size"):
            carpentras_data.solarplot.rolling("ghi", window_size=0)

    def test_invalid_window_size_negative_raises(self, carpentras_data):
        with pytest.raises(ValueError, match="window_size"):
            carpentras_data.solarplot.rolling("ghi", window_size=-1)

    def test_nonexistent_column_raises(self, carpentras_data):
        with pytest.raises(ValueError, match="No valid columns"):
            carpentras_data.solarplot.rolling("nonexistent_xyz")

    def test_invalid_plain_dataframe_raises(self):
        plain = pd.DataFrame({"ghi": [1.0, 2.0, 3.0]})
        with pytest.raises(AttributeError):
            plain.solarplot.rolling()
