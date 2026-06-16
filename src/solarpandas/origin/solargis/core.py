
import importlib
from types import ModuleType

import platformdirs
import pandas as pd
from loguru import logger

from ... import SolarDataFrame
from ...config import get_option
from ...types.annotated import Latitude, Longitude

logger.disable(__name__)
logger = logger.opt(colors=True)


def _import_sg_api_client() -> ModuleType:
    """Import the Solargis API client."""
    try:
        return importlib.import_module("sg_api_client")
    except ModuleNotFoundError as exc:
        msg = ("`sg-api-client` package not available. Please, "
               "install solarpandas as `solarpandas[solargis]`")
        logger.error(msg)
        raise ImportError(msg) from exc


def get_database_path():
    """Get the path to the local BSRN database directory.

    This function retrieves the path from the global configuration. If the
    path is not set, it returns the default path.

    Returns
    -------
    pathlib.Path
        Path to the local BSRN database directory.

    Examples
    --------
    >>> from solarpandas.origin.bsrn import get_database_path
    >>> get_database_path().name
    'bsrn'
    """
    default_path = platformdirs.user_data_path(appname="solarpandas") / "solargis"
    return get_option("solargis.data_dir", default=default_path)


def list_data():
    """List the available Solargis data."""
    path = get_database_path()
    if not path.exists():
        logger.warning(f"Solargis data directory does not exist: {path}")
        return []
    return sorted([f.name for f in path.iterdir() if f.is_file() and f.suffix == ".parquet"])


def list_sgapi_variables():
    """List the available Solargis API variables."""
    sgapi = _import_sg_api_client()
    return sgapi.list_variables()


def list_sgapi_time_steps():
    """List the available Solargis API time steps."""
    sgapi = _import_sg_api_client()
    return sgapi.list_time_steps()


def load_data(
    latitude: Latitude,
    longitude: Longitude,
    years: int | list[int],
    variables: list[str] | None = None,
    time_step: str = "MIN_10",
) -> SolarDataFrame:
    """Load the data from the solarpandas package."""

    if isinstance(years, int):
        year = years

        latstr = f"{'N' if latitude >= 0 else 'S'}{abs(latitude*1e4):07.0f}"
        lonstr = f"{'E' if longitude >= 0 else 'W'}{abs(longitude*1e4):07.0f}"

        path = get_database_path() / f"solargis_{latstr}_{lonstr}_{year}_{time_step}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Loading Solargis data from {path}")

        if not path.exists():
            sgapi = _import_sg_api_client()

            data, metadata = sgapi.request(
                site_lat=latitude,
                site_lon=longitude,
                from_date=f"{year}-01-01",
                to_date=f"{year}-12-31",
                variables=variables or ["GHI", "DNI", "DIF", "TEMP", "WS"],
                time_step=time_step,
                timestamp_alignment=sgapi.TimeAlignment.CENTER,
                terrain_shading=True,
                full_output=True)

            logger.debug(f"Data received for {latitude}, {longitude}")
            logger.debug(f"Metadata received: {metadata}")

            latitude = metadata.pop("latitude", latitude)
            longitude = metadata.pop("longitude", longitude)
            elevation = metadata.pop("elevation", None)

            (
                SolarDataFrame(
                    data,
                    latitude=latitude,
                    longitude=longitude,
                    elevation=elevation,
                    custom_metadata=metadata
                ).to_parquet( path )
            )
            logger.success(f"Solargis data saved to {path}")

        return SolarDataFrame.read_parquet( path )

    return pd.concat([load_data(latitude, longitude, year, variables, time_step) for year in years], axis=0)

