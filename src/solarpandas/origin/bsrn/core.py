
"""Core public API to inspect, fetch, and load BSRN datasets."""

import functools
import gzip
import itertools
import json
import multiprocessing as mp
import re
from datetime import datetime
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, overload

import numpy as np
import pandas as pd
import platformdirs
from loguru import logger

from ...base import SolarDataFrame
from ...iohelpers import read_parquet
from ...config import get_option
from . import helpers, lr_parsers
from .types import DataLogicalRecordName, LogicalRecordName, Month, Site, Year, validate_type

logger.disable(__name__)
logger = logger.opt(colors=True)


SUPPORTED_LOGICAL_RECORDS = [
    # metadata
    "LR0001",
    "LR0002",
    "LR0003",
    "LR0004",
    "LR0005",
    "LR0006",
    "LR0007",
    "LR0008",
    "LR0009",
    # measurements
    "LR0100",  # basic radiation measurements
    "LR0300",  # other radiation measurements (net and upward fluxes)
    "LR0500",  # uv measurements
]


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
    default_path = platformdirs.user_data_path(appname="solarpandas") / "bsrn"
    return get_option("bsrn.data_dir", default=default_path)


def _decode_bsrn_two_digit_year(yy: int, current_year: int | None = None) -> int:
    """Decode BSRN two-digit years into four-digit years.

    Uses a moving pivot based on the current year: values up to current year + 1
    are interpreted as 2000s, the rest as 1900s.
    """
    if current_year is None:
        current_year = datetime.now().year

    pivot = (current_year % 100) + 1
    if yy <= pivot:
        return 2000 + yy
    return 1900 + yy


def _availability_to_year_table(
    availability: dict[str, list[str]], fill_char: str = "#", transposed: bool = False
) -> str:
    """Build a text table of BSRN data availability.

    A filled cell means at least one monthly file is available for that
    site/year; an empty cell means no files for that site/year.

    When ``transposed`` is ``False`` (default), each row is a site and each
    column is a year. The yearly axis uses one character per year with header
    labels every 5 years shown in two rows (YY).

    When ``transposed`` is ``True``, each row is a year (ascending) and each
    column is a site. The site acronyms are displayed vertically in a
    three-row header.
    """
    if len(fill_char) != 1:
        raise ValueError("fill_char must be a single character")

    file_pattern = re.compile(
        r"^(?P<site_dir>[a-zA-Z0-9]{3})/(?P<site_file>[a-zA-Z0-9]{3})(?P<month>\d{2})(?P<yy>\d{2})\.dat\.gz$"
    )

    years_by_site: dict[str, set[int]] = {}
    for site, filenames in availability.items():
        site_years = years_by_site.setdefault(site, set())
        for filename in filenames:
            if not (match := file_pattern.match(filename)):
                continue
            if match.group("site_dir").lower() != match.group("site_file").lower():
                continue
            yy = int(match.group("yy"))
            site_years.add(_decode_bsrn_two_digit_year(yy))

    all_years = sorted({year for years in years_by_site.values() for year in years})
    if not all_years:
        return "No BSRN data availability found."

    if transposed:
        all_sites = sorted(years_by_site.keys())
        year_col_width = 4
        # Three-row header with site acronym written vertically (one char per row)
        header_rows = [
            f"{'':<{year_col_width}} | {''.join(site[i].upper() for site in all_sites)} | {'':<{year_col_width}}"
            for i in range(3)
        ]
        separator = f"{'-' * year_col_width}-+-{'-' * len(all_sites)}-+-{'-' * year_col_width}"
        rows = [
            f"{year} | {''.join(fill_char if year in years_by_site.get(site, set()) else ' ' for site in all_sites)} | {year}"
            for year in all_years
        ]
        return "\n".join(header_rows + [separator] + rows + [separator] + header_rows)

    site_col_width = max(4, max(len(site) for site in years_by_site))

    header_top = []
    header_bottom = []
    for year in all_years:
        if year % 5 == 0:
            yy = f"{year % 100:02d}"
            header_top.append(yy[0])
            header_bottom.append(yy[1])
        else:
            header_top.append(" ")
            header_bottom.append(" ")

    header_row_1 = f"{'site':<{site_col_width}} | {''.join(header_top)}"
    header_row_2 = f"{'':<{site_col_width}} | {''.join(header_bottom)}"
    separator = f"{'-' * site_col_width}-+-{'-' * len(all_years)}"

    rows = []
    repeated_header_block = [header_row_1, header_row_2, separator]
    for i, site in enumerate(sorted(years_by_site)):
        if i > 0 and i % 20 == 0:
            rows.extend(repeated_header_block)
        site_years = years_by_site[site]
        timeline = "".join(fill_char if year in site_years else " " for year in all_years)
        rows.append(f"{site:<{site_col_width}} | {timeline}")

    return "\n".join(repeated_header_block + rows)


def data_availability(
    update: Literal["auto"] | bool = "auto",
    as_year_table: bool = False,
    fill_char: str = "#",
    transposed: bool = False,
    year_table_output: str | Path | None = None,
) -> dict[str, list[str]] | str:
    """Inspect the availability of BSRN data on the remote FTP server.

    This function connects to the BSRN FTP server and retrieves a list of
    available data files for each site. The results are cached locally in a
    JSON file to avoid unnecessary FTP connections. The cache is updated if it
    is older than 7 days or if the `update` parameter is set to `True`.

    Parameters
    ----------
    update : {"auto", bool}, default "auto"
        Whether to refresh the local availability cache. With ``"auto"``, the
        cache is refreshed when older than 7 days.
    as_year_table : bool, default False
        If ``True``, return a plain-text table with one row per site and one
        column per year (or transposed when ``transposed=True``).
    fill_char : str, default "#"
        Character used to mark years with available data in the annual table.
        Must be a single character.
    transposed : bool, default False
        If ``True``, the year table is transposed: each row is a year in
        ascending order and each column is a site. Site acronyms are shown
        vertically in a three-row header. Has no effect when
        ``as_year_table`` is ``False``.
    year_table_output : str or pathlib.Path or None, default None
        Optional output path to persist the annual table as a text file. The
        table is generated when this argument is provided, even if
        ``as_year_table`` is ``False``.

    Returns
    -------
    dict[str, list[str]] or str
        Mapping from site acronym to list of available monthly files, or a
        yearly availability table when ``as_year_table`` is ``True``.

    Examples
    --------
    >>> from solarpandas.origin.bsrn import data_availability
    >>> table = data_availability(update=False, as_year_table=True)
    >>> isinstance(table, str)
    True
    """
    availability_path = get_database_path() / "ftp" / "availability.json"
    file_age_days = helpers.get_file_age(availability_path)

    if update == "auto":
        update = False
        if file_age_days > 7:  # update if the file is older than 7 days
            logger.info(
                f"Availability file is {file_age_days:.1f} days old. Updating..."
            )
            update = True

    if update or not availability_path.exists():
        availability = helpers.inspect_data_availability(timeout=30)
        availability_path.parent.mkdir(parents=True, exist_ok=True)
        with availability_path.open("w") as f:
            json.dump(availability, f, indent=4)

    with availability_path.open("r") as f:
        availability = json.load(f)

    year_table = None
    if as_year_table or year_table_output is not None:
        year_table = _availability_to_year_table(availability, fill_char=fill_char, transposed=transposed)

    if year_table_output is not None:
        output_path = Path(year_table_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(year_table, encoding="utf-8")

    if as_year_table:
        return year_table
    return availability


def load_metadata(update: Literal["auto"] | bool = "auto"):
    """Load cached station metadata, optionally refreshing remote source.

    Parameters
    ----------
    update : {"auto", bool}, default "auto"
        If ``True``, force refresh from PANGAEA. If ``"auto"``, refresh when
        cache age is older than 7 days.

    Returns
    -------
    dict[str, Any]
        Station metadata dictionary keyed by site acronym.

    Examples
    --------
    >>> from solarpandas.origin.bsrn import load_metadata
    >>> meta = load_metadata(update=False)
    >>> isinstance(meta, dict)
    True
    """

    metadata_path = get_database_path() / "ftp" / "metadata.json"
    file_age_days = helpers.get_file_age(metadata_path)

    if update == "auto":
        update = False
        if file_age_days > 7:  # update if the file is older than 7 days
            logger.info(f"Metadata file is {file_age_days:.1f} days old. Updating...")
            update = True

    if update or not metadata_path.exists():
        metadata = helpers.fetch_allsite_metadata_from_pangaea()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=4)

    with metadata_path.open("r") as f:
        return json.load(f)


def load_data(
    site: Site,
    years: Sequence[Year] | Year,
    logical_record: Literal["LR0100", "LR0300", "LR0500"] = "LR0100",
    group: Literal["essential", "avg", "all"] = "essential",
) -> SolarDataFrame | None:
    """Load yearly BSRN data from cache or raw FTP files.

    Parameters
    ----------
    site : Site
        Three-letter BSRN station code.
    years : Year or sequence of Year
        Year or list of years to retrieve.
    logical_record : {"LR0100", "LR0300", "LR0500"}, default "LR0100"
        Logical record to load and cache.
    group : {"essential", "avg", "all"}, default "essential"
        Variable group selection based on CF metadata tags.

    Returns
    -------
    SolarDataFrame or None
        Combined dataframe with harmonized metadata, or ``None`` when no data
        could be retrieved.

    Examples
    --------
    >>> from solarpandas.origin.bsrn import load_data
    >>> sdf = load_data(site="car", years=2016)
    >>> sdf is None or "ghi" in sdf.columns
    True
    """

    site = validate_type(site, Site)
    years = [validate_type(year, Year) for year in np.asarray(years, dtype=int).reshape(-1)]
    logical_record = validate_type(logical_record, DataLogicalRecordName)

    # For standard_names see: https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html
    if not (path := files("solarpandas").joinpath("origin/bsrn/cf-metadata.json")).exists():
        logger.warning("CF metadata file not found. Cannot load metadata.")
    cf_metadata = json.loads(path.read_text())

    def collect_variables_metadata(columns: list[str]) -> dict:
        var_metadata ={}
        for varname in columns:
            if varname in cf_metadata:
                values = {key: value for key, value in cf_metadata[varname].items() if not key.startswith("_")}
                var_metadata[values["short_name"]] = values | {"bsrn_name": varname}
            else:
                logger.warning(f"No CF metadata found for variable '{varname}'. Skipping metadata assignment for this variable.")
                var_metadata[varname] = {
                    "standard_name": "unknown",
                    "long_name": "unknown",
                    "short_name": "unknown",
                    "units": "unknown",
                    "bsrn_name": varname,
                }
        return var_metadata

    def filter_columns(columns: list[str], group: str) -> list[str]:
        logger.debug(f"Filtering columns for group '{group}'...")

        group_columns = []

        # 1. Take all group columns in cf-metadata
        if group.casefold() == "all":
            group_columns = [cf_metadata.get(column, {}).get("short_name", column) for column in columns]
        else:
            for vattrs in cf_metadata.values():
                if "_groups" not in vattrs:
                    continue
                if group in map(str.strip, vattrs["_groups"].split(",")):
                    group_columns.append(vattrs["short_name"])

            logger.debug(f"Group columns: {group_columns}")

        # 2. Take the intersection with the columns in the data
        return [col for col in columns if col in group_columns]

    db_path = get_database_path() / "cached" / site
    db_path.mkdir(parents=True, exist_ok=True)

    load_bsrn_files = functools.partial(
        load_data_from_bsrn_files,
        site=site,
        months=range(1, 13),
        filled=True,
        centered=True,
        include_metadata=False,
        extra_records=None if logical_record == "LR0100" else [logical_record])

    paths = []
    for year in years:
        if not (file_path := db_path / f"{site}_{year}_{logical_record.lower()}.parquet").exists():
            logger.info(f"cached file {file_path.name} not found. Loading data from BSRN files...")

            # 1. collect the data
            data = load_bsrn_files(years=year)
            if logical_record != "LR0100":
                data = data[1][logical_record]  # logical_record is in [LR0300, LR0500]
            if data is None:
                logger.warning(f"no data retrieved for {site=}, {year=}, and {logical_record=}. Skipping...")
                continue

            # 2. collect variables metadata
            vmetadata = collect_variables_metadata(data.columns.tolist())
            data.custom_metadata["variables"] = vmetadata.copy()
            rename_map = {meta["bsrn_name"]: var for var, meta in vmetadata.items()}

            # 3. update custom_metadata and column names and save to parquet
            (
                data
                .rename(columns=rename_map)
                .rename_axis("time", axis=0)
                .astype(np.float32)
                .reset_index()
                .to_parquet(file_path)
            )
        paths.append(file_path)

    if not paths:
        logger.warning(f"no data available for {site=}, {years=}, and {logical_record=}. Returning None.")
        return None

    unfiltered_data = pd.concat([read_parquet(path) for path in sorted(paths)], axis=0).set_index("time")
    selected_columns = filter_columns(unfiltered_data.columns.tolist(), group=group)
    data = unfiltered_data.get(selected_columns)
    data.custom_metadata["variables"] = {var: meta for var, meta in data.custom_metadata["variables"].items()
                                         if var in selected_columns}
    return data


def clear_cache(
    site: Site | None = None,
    year: Year | None = None,
    logical_record: DataLogicalRecordName | None = None
) -> None:
    """Clear cached data files for a given site, year and logical record.

    Parameters
    ----------
    site : Site or None, default None
        Three-letter station code. If ``None``, all sites are included.
    year : Year or None, default None
        Year to clear. If ``None``, all years are included.
    logical_record : DataLogicalRecordName or None, default None
        Logical record to clear. If ``None``, all logical records are included.

    Examples
    --------
    >>> from solarpandas.origin.bsrn import clear_cache
    >>> clear_cache(site="car", year=2016, logical_record="LR0100")
    """

    cache_path = get_database_path() / "cached"

    site_pattern = validate_type(site, Site) if site is not None else "*"
    sites_to_clear = list(cache_path.glob(site_pattern))
    logger.debug(f"Sites to clear: {[site.name for site in sites_to_clear]}")

    year = validate_type(year, Year)
    year_pattern = f"{year}" if year is not None else "*"

    logical_record = validate_type(logical_record, DataLogicalRecordName)
    lr_pattern = logical_record.lower() if logical_record is not None else "*"

    for site_dir in sites_to_clear:
        path = f"{site_dir.name}_{year_pattern}_{lr_pattern}.parquet"
        files_to_clear = list(site_dir.glob(path))
        logger.debug(f"Files to clear: {files_to_clear}")
        for file in files_to_clear:
            file.unlink()
            logger.info(f"Cleared cached file: {file}")
        if not list(site_dir.iterdir()):
            site_dir.rmdir()
            logger.info(f"Removed empty site directory: {site_dir}")


def __parse_bsrn_file__(site, year, month, logical_records = None):
    set_of_lr = set(["LR0004", "LR0100"] + (logical_records if logical_records is not None else []))
    retrieval = parse_bsrn_file(
        get_database_path() / "ftp" / site / f"{site}{month:02d}{str(year)[-2:]}.dat.gz",
        check_remote_on_missing_file=True,
        logical_records=list(set_of_lr),
        timeout=30)
    return retrieval | {"year": year, "month": month}


# CASE 1: only data
@overload
def load_data_from_bsrn_files(
    site: Site,
    years: Sequence[Year] | Year,
    months: Sequence[Month] | Month = range(1, 13),
    filled: bool = True,
    centered: bool = True,
    include_metadata: Literal[False] = False,
    extra_records: None = None,
) -> None | SolarDataFrame: ...


# CASE 2: data and metadata
@overload
def load_data_from_bsrn_files(
    site: Site,
    years: Sequence[Year] | Year,
    months: Sequence[Month] | Month = range(1, 13),
    filled: bool = True,
    centered: bool = True,
    include_metadata: Literal[True] = True,
    extra_records: None = None,
) -> None |tuple[SolarDataFrame, pd.DataFrame]: ...


# CASE 3: data and extra data
@overload
def load_data_from_bsrn_files(
    site: Site,
    years: Sequence[Year] | Year,
    months: Sequence[Month] | Month = range(1, 13),
    filled: bool = True,
    centered: bool = True,
    include_metadata: Literal[False] = False,
    extra_records: list[Literal["LR0300", "LR0500"]] = ...,
) -> None | tuple[SolarDataFrame, dict[str, SolarDataFrame]]: ...


# CASE 4: data, metadata and extra data
@overload
def load_data_from_bsrn_files(
    site: Site,
    years: Sequence[Year] | Year,
    months: Sequence[Month] | Month = range(1, 13),
    filled: bool = True,
    centered: bool = True,
    include_metadata: Literal[True] = True,
    extra_records: list[Literal["LR0300", "LR0500"]] = ...,
) -> None | tuple[SolarDataFrame, pd.DataFrame, dict[str, SolarDataFrame]]: ...


def load_data_from_bsrn_files(
    site: Site,
    years: Sequence[Year] | Year,
    months: Sequence[Month] | Month = range(1, 13),
    filled: bool = True,
    centered: bool = True,
    include_metadata: bool = False,
    extra_records: list[Literal["LR0300", "LR0500"]] | None = None,
):
    """Load and parse monthly BSRN ``.dat.gz`` files from local FTP mirror.

    Parameters
    ----------
    site : Site
        Three-letter station code.
    years : Year or sequence of Year
        Years to load.
    months : Month or sequence of Month, default ``range(1, 13)``
        Months to load.
    filled : bool, default True
        If ``True``, reindex to dense 1-minute frequency.
    centered : bool, default True
        If ``True``, shift timestamps by 30 seconds to represent minute centers.
    include_metadata : bool, default False
        Whether to include per-file metadata dataframe in output tuple.
    extra_records : list[{"LR0300", "LR0500"}] or None, default None
        Additional logical records to parse and return.

    Returns
    -------
    Various
        Return type depends on ``include_metadata`` and ``extra_records``
        according to overload declarations above.
    """

    site = validate_type(site, Site)
    years = [validate_type(year, Year) for year in np.asarray(years, dtype=int).reshape(-1)]
    months = [validate_type(month, Month) for month in np.asarray(months, dtype=int).reshape(-1)]

    if extra_records is not None:
        for lr in extra_records:
            if lr not in ["LR0300", "LR0500"]:
                raise ValueError(f"invalid logical record name in extra_records: {lr}. "
                                 "Supported values are 'LR0300' and 'LR0500'.")

    parse_bsrn_file_with_extra_records = functools.partial(__parse_bsrn_file__, logical_records=extra_records)

    list_of_years_and_months = sorted(itertools.product(years, months), key=lambda x: (x[0], x[1]))

    logger.info(f"loading data for {len(list_of_years_and_months)} BSRN files...")

    tasks = [(site, year, month) for year, month in list_of_years_and_months]
    with mp.Pool(mp.cpu_count()) as workers:
        # starmap keeps the order of the tasks, so the output is ordered by year and month
        retrievals = workers.starmap(parse_bsrn_file_with_extra_records, tasks, chunksize=1)

    # remove empty retrievals (files not found or with no supported logical records)
    def is_missing(retrieval: dict) -> bool:
        if not retrieval:
            return True
        if "LR0100" not in retrieval or retrieval["LR0100"].empty:
            year = retrieval.get("year")
            month = retrieval.get("month")
            logger.warning(f"no data retrieved for {site=}, {year=}, and {month=}")
            return True
        return False
    if not (retrievals := [retr for retr in retrievals if not is_missing(retr)]):
        logger.warning(f"no data retrieved for {site=}, {years=}, and {months=}")
        return None

    #===================================================================================
    # PREPARE THE DATA AND METADATA TO BE RETURNED.
    #   DATA IS A SOLARDATAFRAME
    #   METADATA IS A PANDAS DATAFRAME
    # The metadata included in the data solardataframe, included latitude,
    # longitude and altitude are gathered from `load_metadata`, which retrieves
    # them from Pangaea. The metadata included in the metadata dataframe, included
    # surface type, topography type, horizon azimuth and elevation, are gathered
    # from the logical record LR0004 of each file. The metadata included in the
    # data solardataframe is expected to be consistent across all files, while the
    # metadata included in the metadata dataframe may vary across files (e.g., if
    # there are changes in the surface type or topography type during the period
    # of interest).
    #===================================================================================

    def clean_data_retrieval(retrieval: dict, lr: LogicalRecordName) -> pd.DataFrame:
        if (data := retrieval.get(lr)) is None:
            return None

        year = retrieval.get("year")
        month = retrieval.get("month")

        # set a DatetimeIndex with the time information in the logical record
        time_dict = {"year": year, "month": month, "day": data["day"], "hour": data["hour"], "minute": data["minute"]}
        times_utc = pd.to_datetime(pd.DataFrame(time_dict), utc=True)
        data = data.set_index(times_utc).drop(columns=["day", "hour", "minute"])

        # add missing timestamps with NaN values, if necessary
        start = pd.to_datetime(f"{year}-{month:02d}-01")
        end = start + pd.offsets.MonthBegin(1)
        dense_times = pd.date_range(start, end, freq="1min", inclusive="left", tz="UTC")
        return data.reindex(dense_times)

    data = pd.concat([clean_data_retrieval(retr, lr="LR0100") for retr in retrievals], axis=0)

    if extra_records is not None:
        extra_data = {}
        for lr in extra_records:
            logger.info(f"processing extra logical record {lr}...")
            clean_retrievals = [clean_data_retrieval(retr, lr=lr) for retr in retrievals]
            if not len(this_data := [df for df in clean_retrievals if df is not None]):
                logger.warning(f"no data retrieved for logical record {lr} in {site=}, {years=}, and {months=}")
                extra_data[lr] = None
            else:
                extra_data[lr] = pd.concat(this_data, axis=0)

    if centered:
        data = data.set_index(data.index + pd.to_timedelta("30s"))
        if extra_records is not None:
            for lr, df in extra_data.items():
                if df is not None:
                    extra_data[lr] = df.set_index(df.index + pd.to_timedelta("30s"))

    if filled:
        dense_times = pd.date_range(data.index.min(), data.index.max(), freq="1min", inclusive="both", tz="UTC")
        data = data.reindex(dense_times)
        if extra_records is not None:
            for lr, df in extra_data.items():
                if df is not None:
                    dense_times = pd.date_range(df.index.min(), df.index.max(), freq="1min", inclusive="both", tz="UTC")
                    extra_data[lr] = df.reindex(dense_times)

    variables = ("surface_type", "topography_type", "latitude", "longitude",
                 "altitude", "horizon_azimuth", "horizon_elevation")
    metadata = [{"year": retr["year"], "month": retr["month"]} |
                {key: retr["LR0004"][key] for key in variables}
                for retr in retrievals]
    metadata = pd.DataFrame.from_records(metadata)

    if metadata["latitude"].nunique() > 1:
        logger.warning("the retrieved data contains different latitude values "
                       f"({metadata['latitude'].unique()}). This is not expected.")

    if metadata["longitude"].nunique() > 1:
        logger.warning("the retrieved data contains different longitude values "
                       f"({metadata['longitude'].unique()}). This is not expected.")

    if metadata["altitude"].nunique() > 1:
        logger.warning("the retrieved data contains different altitude values "
                       f"({metadata['altitude'].unique()}). This is not expected.")

    allsite_metadata = load_metadata()
    site_metadata = allsite_metadata.get(site.casefold(), {})

    latitude = site_metadata.get("latitude", metadata["latitude"].iloc[-1] if "latitude" in metadata else None)
    if latitude is None:
        raise ValueError("latitude is missing.")

    longitude = site_metadata.get("longitude", metadata["longitude"].iloc[-1] if "longitude" in metadata else None)
    if longitude is None:
        raise ValueError("longitude is missing.")

    elevation = site_metadata.get("altitude", metadata["altitude"].iloc[-1] if "altitude" in metadata else None)
    if elevation is None:
        logger.warning("elevation is missing. Setting elevation to 0.")

    custom_metadata = {}
    custom_metadata["station"] = site.upper()
    custom_metadata["location"] = site_metadata.get("station", "unknown")
    if "location" in site_metadata:
        province_and_or_country = site_metadata["location"]
        custom_metadata["location"] = custom_metadata["location"] + f", {province_and_or_country}"
    custom_metadata["network"] = "BSRN"
    custom_metadata["source"] = "BSRN FTP server via solarpandas"
    custom_metadata["institution"] = "Jose A Ruiz-Arias (solarpandas dev) and BSRN data providers"
    custom_metadata["contact"] = "jararias@uma.es"

    custom_metadata["timestamp_alignment"] = "center" if centered else "start"
    custom_metadata["surface_type"] = metadata["surface_type"].iloc[-1] if "surface_type" in metadata else "unknown"
    custom_metadata["topography_type"] = metadata["topography_type"].iloc[-1] if "topography_type" in metadata else "unknown"
    custom_metadata["horizon_azimuth"] = metadata["horizon_azimuth"].iloc[-1] if "horizon_azimuth" in metadata else "unknown"
    custom_metadata["horizon_elevation"] = metadata["horizon_elevation"].iloc[-1] if "horizon_elevation" in metadata else "unknown"

    data = SolarDataFrame(
        data,
        latitude=latitude,
        longitude=longitude,
        elevation=elevation,
        custom_metadata=custom_metadata)

    if extra_records is not None:
        for lr, df in extra_data.items():
            if df is not None:
                extra_data[lr] = SolarDataFrame(
                    df,
                    latitude=latitude,
                    longitude=longitude,
                    elevation=elevation,
                    custom_metadata=custom_metadata)

    if not include_metadata:
        if extra_records is None:
            return data  # overload case 1
        return data, extra_data  # overload case 3

    if extra_records is None:
        return data, metadata  # overload case 2
    return data, metadata, extra_data  # overload case 4


@dataclass
class LogicalRecord:
    """Representation of one logical record block found in a BSRN file."""

    signature: str
    first_line: int
    last_line: int
    lines: list[str]
    parser: Callable | None = None

    def __post_init__(self):
        if not re.match(r"^\*[CU]\d\d\d\d$", self.signature):
            raise ValueError(f"invalid logical record signature: `{self.signature}`")
        if self.first_line < 0 or self.last_line < self.first_line:
            raise ValueError(
                f"invalid logical record line numbers: {self.first_line}, {self.last_line}"
            )
        if self.lines[0] != self.signature:
            raise ValueError(
                f"the first line of the logical record does not match the signature: "
                f"{self.lines[0]} != {self.signature}"
            )
        self.lines = self.lines[1:]

        if hasattr(lr_parsers, f"parse_logical_record_{self.id}"):
            self.parser = getattr(lr_parsers, f"parse_logical_record_{self.id}")

    @property
    def id(self):
        return self.signature[2:]

    @property
    def name(self):
        return f"LR{self.id}"

    @property
    def has_changed(self):
        return self.signature[1] == "C"

    def parse(self, **kwargs) -> dict[str, Any]:
        """Parse record lines with the associated logical-record parser."""
        if self.parser is None:
            raise ValueError("no parser available for logical record {self.name}")
        logger.debug(f"parsing <blue>{self.name}</blue> with parser <blue>{self.parser.__name__}</blue>")
        return self.parser(self.lines, **kwargs)

    @classmethod
    def find_in_data(
        cls, txt_data: list[str], supported: list[str] | None = None
    ) -> list["LogicalRecord"]:
        regex = re.compile(r"^\s*\*[CU]\d\d\d\d\s*$")
        indices_gen = (i for i, line in enumerate(txt_data) if regex.match(line))  # noqa: F506
        lr_start = list(indices_gen)
        lr_end = [i - 1 for i in lr_start[1:]] + [len(txt_data)]
        logical_records = [
            cls(
                signature=txt_data[i_start].strip(),
                first_line=i_start,
                last_line=i_end,
                lines=txt_data[i_start : i_end + 1],
            )
            for i_start, i_end in zip(lr_start, lr_end)
        ]
        if supported is not None:
            return [lr for lr in logical_records if lr.name in supported]
        return logical_records


def parse_bsrn_file(
    path: Path,
    check_remote_on_missing_file: bool = True,
    logical_records: LogicalRecordName | list[LogicalRecordName] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Parse selected logical records from a BSRN ``.dat.gz`` file.

    Parameters
    ----------
    path : pathlib.Path
        Local path to monthly BSRN file.
    check_remote_on_missing_file : bool, default True
        If ``True``, attempt remote FTP download when file is missing locally.
    logical_records : LogicalRecordName or list[LogicalRecordName] or None
        Records to parse. When ``None``, all supported records in file are parsed.
    timeout : int, default 30
        FTP timeout in seconds for remote retrieval.

    Returns
    -------
    dict[str, Any]
        Parsed content keyed by logical record name (e.g. ``"LR0100"``).
    """

    if logical_records is not None:
        if isinstance(logical_records, str):
            logical_records = [logical_records]
        logical_records = [
            validate_type(lr, LogicalRecordName) for lr in logical_records
        ]

    if logical_records is None:
        logger.debug("the user has not specified a list of logical records to parse")
    else:
        logger.debug(
            f"the user has specified the following logical records to parse: {logical_records}"
        )

    logger.debug(f"the supported logical records are: {SUPPORTED_LOGICAL_RECORDS}")

    if check_remote_on_missing_file and not path.exists():
        try:
            path = helpers.fetch_site_data_from_ftp(
                path.name, path.parent, timeout=timeout
            )
        except Exception as exc:
            logger.warning(exc)

    if not path.exists():
        logger.error(
            f"BSRN data file {path.name} not found in {path.parent}"
        )
        return {}

    logger.info(f"reading file <blue>{path.name}</blue> (@ {path.parent})")
    with gzip.open(path, "rb") as gz:
        txt_data = [line.rstrip().decode("utf-8") for line in gz.readlines()]

    # find all logical records in the data but keep only the supported ones (if specified)
    logical_records_in_data = LogicalRecord.find_in_data(txt_data)

    logger.debug(
        f"the following logical records are present in the data: {[lr.name for lr in logical_records_in_data]}"
    )

    supported_logical_records_in_data = [
        lr for lr in logical_records_in_data if lr.name in SUPPORTED_LOGICAL_RECORDS
    ]

    # if the user does not specify which logical records to parse, parse all the supported ones
    # that are present in the data.
    if logical_records is None:
        logical_records = supported_logical_records_in_data

    # otherwise, parse only the user specified logical records that are present in the data and
    # raise a warning to the user about the rest.
    else:
        logical_records_to_be_parsed = []
        for lr in logical_records:
            if lr in [lr_.name for lr_ in supported_logical_records_in_data]:
                logical_records_to_be_parsed.append(lr)
            else:
                if lr in SUPPORTED_LOGICAL_RECORDS:
                    logger.warning(f"the logical record {lr} is not in data. Ignoring it.")
                else:
                    logger.warning(f"the logical record {lr} is not supported. Ignoring it.")
        logical_records = [
            lr
            for lr in supported_logical_records_in_data
            if lr.name in logical_records_to_be_parsed
        ]

    logger.debug(
        f"the following logical records will be parsed: {[lr.name for lr in logical_records]}"
    )

    contents = {}

    for logical_record in sorted(logical_records, key=lambda lr: lr.name):  # sort logical records by their name (LRxxxx)
        if not logical_record.parser:
            logger.warning(
                f"unavailable parser for logical record with id {logical_record.name}"
            )
            continue

        lr_data = logical_record.parse(path=path)
        contents[logical_record.name] = lr_data

    return contents
