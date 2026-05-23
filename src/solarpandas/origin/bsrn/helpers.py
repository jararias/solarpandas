
import re
import ftplib
import json
from datetime import datetime, timezone
import numpy as np
from pathlib import Path
from netrc import netrc

import pandas as pd
from loguru import logger

from ...config import get_option
from .types import Site, validate_type

logger.disable(__name__)
logger = logger.opt(colors=True)


def get_file_age(path: Path):
    if not path.exists():
        return np.inf
    datetime_created = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    file_age = datetime.now(timezone.utc) - datetime_created
    return file_age.total_seconds() / (24 * 3600)  # seconds to days


def fetch_site_data_from_ftp(
    remote_fn: str,
    local_path: str | Path,  # directory to store the downloaded file
    user: str | None = None,
    password: str | None = None,
    timeout: int | None = None
) -> None:

    site = validate_type(remote_fn[:3], Site)

    if isinstance(local_path, str):
        local_path = Path(local_path)

    if (server := get_option("bsrn.server", None)) is None:
        raise ValueError("BSRN server not specified in config file")

    if user is None or password is None:
        logger.debug("User or password not provided. Using credentials from netrc file")
        if (retrieval := netrc().authenticators(server)) is None:
            raise ValueError(f"credentials for server `{server}` not found in netrc file")
        user = user or retrieval[0]
        password = password or retrieval[2]

    try:
        # open the FTP connection and log in with the provided credentials
        with ftplib.FTP(server, timeout=timeout) as ftp:
            ftp.login(user, password)

            # check if there is a directory for this site on the server
            if site not in list(filter(lambda x: len(x) == 3, ftp.nlst())):
                raise ValueError(f"site `{site}` not found on BSRN server")
            ftp.cwd(site)  # change to site directory

            # check if the requested file is available for download
            if remote_fn not in list(filter(lambda x: x.endswith("dat.gz"), ftp.nlst())):
                raise ValueError(f"file `{remote_fn}` not found on BSRN server")

            # prepare the local path to download the file
            local_path.mkdir(parents=True, exist_ok=True)

            # download the file using FTP's RETR command
            logger.info(f"downloading file `{remote_fn}` from BSRN server")
            with (local_path / remote_fn).open("wb") as f:
                ftp.retrbinary(f"RETR {remote_fn}", f.write)
            logger.success(f"<blue>{remote_fn}</blue> added to {local_path}")

    except ftplib.all_errors as exc:
        # catch login, connection, and other FTP-related errors
        raise ValueError(f"loging error: {exc.args[0]}") from exc

    except OSError as exc:
        # catch network-related errors (e.g., DNS failure, refused connection)
        raise ValueError(f"network error: {exc.strerror}") from exc

    return local_path / remote_fn

def fetch_allsite_metadata_from_pangaea():
    # More tables in:
    #  https://dataportals.pangaea.de/bsrn/
    #  End of: hhttps://bsrn.awi.de/data/data-retrieval-via-pangaea/

    url = "https://www.pangaea.de/ddi?request=bsrn/BSRNEvent&format=html&title=BSRN+Stations"
    logger.debug(f"fetching BSRN station metadata from {url}")
    table = pd.read_html(url)[0]

    column_mapping = {
        "Event, optional label": "station",
        "Event label": "acronym",
        "Location": "location",
        "Station info": "info",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Elevation": "altitude",
        "Date/Time start": "start",
        "Date/Time end": "end",
        "Comment": "comment",
        "URI of event": "uri",
    }

    table = table.rename(columns=column_mapping)
    table = table[list(column_mapping.values())]
    table["acronym"] = table["acronym"].str.strip().str.lower()
    table["latitude"] = table["latitude"].astype(float)
    table["longitude"] = table["longitude"].astype(float)
    table["altitude"] = table["altitude"].astype(float)
    table["start"] = pd.to_datetime(table["start"], errors="coerce")
    table["end"] = pd.to_datetime(table["end"], errors="coerce")

    records = json.loads(table.to_json(orient="records"))
    return {record["acronym"]: {key: value for key, value in record.items() if key != "acronym"}
            for record in records}


def inspect_data_availability(
    user: str | None = None,
    password: str | None = None,
    timeout: int | None = 30,
) -> dict[str, list[str]]:

    if (server := get_option("bsrn.server", None)) is None:
        raise ValueError("BSRN server not specified in config file")

    if user is None or password is None:
        logger.debug("User or password not provided. Using credentials from netrc file")
        if (retrieval := netrc().authenticators(server)) is None:
            raise ValueError(f"credentials for server `{server}` not found in netrc file")
        user = user or retrieval[0]
        password = password or retrieval[2]

    try:
        # open the FTP connection and log in with the provided credentials
        with ftplib.FTP(server, timeout=timeout) as ftp:
            ftp.login(user, password)

            sites = {}
            for site in sorted(filter(lambda x: len(x) == 3, ftp.nlst())):
                regex = re.compile(r"^{0}/{0}\d{{4}}\.dat\.gz$".format(site))
                files = sorted(list(filter(regex.match, ftp.nlst(site))))
                logger.info(f"<blue>{site=}</blue>: {len(files)} files available")
                sites[site] = files

    except ftplib.all_errors as exc:
        # catch login, connection, and other FTP-related errors
        raise ValueError(f"loging error: {exc.args[0]}") from exc

    except OSError as exc:
        # catch network-related errors (e.g., DNS failure, refused connection)
        raise ValueError(f"network error: {exc.strerror}") from exc

    return sites
