import matplotlib

matplotlib.use("Agg")  # headless backend — must be set before pyplot is imported

import numpy as np
import pandas as pd
import pytest

import solarpandas as sp

CAR_LAT = 44.083
CAR_LON = 5.059
CAR_ELV = 100.0

DAY = "2016-06-21"
N_MINUTES = 1440


@pytest.fixture(scope="session")
def location():
    return {"latitude": CAR_LAT, "longitude": CAR_LON, "elevation": CAR_ELV}


@pytest.fixture(scope="session")
def times_1day():
    return pd.date_range(DAY, periods=N_MINUTES, freq="1min", tz="UTC")


@pytest.fixture(scope="session")
def solar_series(times_1day, location):
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 500, size=N_MINUTES)
    return sp.SolarSeries(data, index=times_1day, name="ghi", **location)


@pytest.fixture(scope="session")
def solar_dataframe(times_1day, location):
    rng = np.random.default_rng(42)
    return sp.SolarDataFrame(
        {
            "ghi": rng.uniform(0, 500, N_MINUTES),
            "dni": rng.uniform(0, 800, N_MINUTES),
            "dif": rng.uniform(0, 200, N_MINUTES),
        },
        index=times_1day,
        **location,
    )


@pytest.fixture(scope="session")
def carpentras_data():
    from solarpandas.sample_data import load_carpentras_data

    data = load_carpentras_data()
    day = data.loc[DAY]
    # Localize tz-naive index to UTC so solpos accessor works
    if day.index.tz is None:
        day = day.__class__(
            day,
            latitude=day.latitude,
            longitude=day.longitude,
            elevation=day.elevation,
            custom_metadata=day.custom_metadata.copy(),
        )
        day.index = day.index.tz_localize("UTC")
    return day
