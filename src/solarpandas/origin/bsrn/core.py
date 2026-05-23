
import functools
import gzip
import itertools
import json
import multiprocessing as mp
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np
import pandas as pd
import platformdirs
from loguru import logger

from ...base import SolarDataFrame
from ...config import get_option
from . import helpers, lr_parsers
from .types import LogicalRecordName, Month, Site, Year, validate_type

logger.disable(__name__)
logger = logger.opt(colors=True)


SUPPORTED_LOGICAL_RECORDS = [
    "LR0001",
    "LR0002",
    "LR0003",
    "LR0004",
    "LR0005",
    "LR0006",
    "LR0007",
    "LR0008",
    "LR0009",
    "LR0100",
    "LR0300",
    "LR0500",
]


MAPPING_OF_NAMES = {
    'global_horizontal_avg': {'short_name': 'ghi', 'description': 'global horizontal irradiance', 'unit': 'W m-2'},
    'global_horizontal_std': {'short_name': 'ghi_std', 'description': 'standard deviation of global horizontal irradiance', 'unit': 'W m-2'},
    'global_horizontal_min': {'short_name': 'ghi_min', 'description': 'minimum global horizontal irradiance', 'unit': 'W m-2'},
    'global_horizontal_max': {'short_name': 'ghi_max', 'description': 'maximum global horizontal irradiance', 'unit': 'W m-2'},
    'direct_normal_avg': {'short_name': 'dni', 'description': 'direct normal irradiance', 'unit': 'W m-2'},
    'direct_normal_std': {'short_name': 'dni_std', 'description': 'standard deviation of direct normal irradiance', 'unit': 'W m-2'},
    'direct_normal_min': {'short_name': 'dni_min', 'description': 'minimum direct normal irradiance', 'unit': 'W m-2'},
    'direct_normal_max': {'short_name': 'dni_max', 'description': 'maximum direct normal irradiance', 'unit': 'W m-2'},
    'diffuse_horizontal_avg': {'short_name': 'dif', 'description': 'diffuse horizontal irradiance', 'unit': 'W m-2'},
    'diffuse_horizontal_std': {'short_name': 'dif_std', 'description': 'standard deviation of diffuse horizontal irradiance', 'unit': 'W m-2'},
    'diffuse_horizontal_min': {'short_name': 'dif_min', 'description': 'minimum diffuse horizontal irradiance', 'unit': 'W m-2'},
    'diffuse_horizontal_max': {'short_name': 'dif_max', 'description': 'maximum diffuse horizontal irradiance', 'unit': 'W m-2'},
    'downward_longwave_avg': {'short_name': 'lwd', 'description': 'downward longwave irradiance', 'unit': 'W m-2'},
    'downward_longwave_std': {'short_name': 'lwd_std', 'description': 'standard deviation of downward longwave irradiance', 'unit': 'W m-2'},
    'downward_longwave_min': {'short_name': 'lwd_min', 'description': 'minimum downward longwave irradiance', 'unit': 'W m-2'},
    'downward_longwave_max': {'short_name': 'lwd_max', 'description': 'maximum downward longwave irradiance', 'unit': 'W m-2'},
    'air_temperature': {'short_name': 'temp', 'description': 'air temperature', 'unit': '°C'},
    'relative_humidity': {'short_name': 'rh', 'description': 'relative humidity', 'unit': '%'},
    'atmospheric_pressure': {'short_name': 'pres', 'description': 'atmospheric pressure', 'unit': 'hPa'}
}


def get_database_path():
    """Get the path to the local BSRN database directory.

    This function retrieves the path from the global configuration. If the
    path is not set, it returns the default path.

    Returns:
        Path | None: The path to the local BSRN database directory, or `None`
            if it is not set in the configuration. If the path is set, it is returned as a `Path` object.
    """
    default_path = platformdirs.user_data_path(appname="solarpandas") / "bsrn"
    return get_option("bsrn.data_dir", default=default_path)


def data_availability(update: Literal["auto"] | bool = "auto") -> dict[str, list[str]]:
    """Inspect the availability of BSRN data on the remote FTP server.

    This function connects to the BSRN FTP server and retrieves a list of
    available data files for each site. The results are cached locally in a
    JSON file to avoid unnecessary FTP connections. The cache is updated if it
    is older than 7 days or if the `update` parameter is set to `True`.

    Args:
        update (Literal["auto"] | bool, optional): Whether to update the local cache of data availability. If set to "auto", the cache will be updated if it is older than 7 days. Defaults to "auto".

    Returns:
        dict[str, list[str]]: A dictionary where keys are site identifiers and values are lists of available data files for each site.
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
        return json.load(f)


def load_metadata(update: Literal["auto"] | bool = "auto"):

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


def __parse_bsrn_file__(site, year, month, logical_records = None):
    retrieval = parse_bsrn_file(
        get_database_path() / "ftp" / site / f"{site}{month:02d}{str(year)[-2:]}.dat.gz",
        check_remote_on_missing_file=True,
        logical_records=["LR0004", "LR0100"] + (logical_records if logical_records is not None else []),
        timeout=30)
    return retrieval | {"year": year, "month": month}


def load_data(
    site: Site,
    years: Sequence[Year] | Year,
    months: Sequence[Month] | Month = range(1, 13),
    filled: bool = True,
    centered: bool = True,
    reduced: bool = True,
    extra_output: list[Literal["LR0300", "LR0500"]] | None = None,
) -> tuple[SolarDataFrame, pd.DataFrame] | tuple[SolarDataFrame, pd.DataFrame, dict[str, SolarDataFrame]]:

    # TODO: remapear los nombres y seleccionar por defecto solo las mas relevantes

    site = validate_type(site, Site)
    years = [validate_type(year, Year) for year in np.asarray(years, dtype=int).reshape(-1)]
    months = [validate_type(month, Month) for month in np.asarray(months, dtype=int).reshape(-1)]

    if extra_output is not None:
        for lr in extra_output:
            if lr not in ["LR0300", "LR0500"]:
                raise ValueError(f"invalid logical record name in extra_output: {lr}. Supported values are 'LR0300' and 'LR0500'.")

    parse_bsrn_file_with_extra_records = functools.partial(__parse_bsrn_file__, logical_records=extra_output)

    list_of_years_and_months = sorted(itertools.product(years, months), key=lambda x: (x[0], x[1]))

    logger.info(f"loading data for {len(list_of_years_and_months)} BSRN files...")

    tasks = [(site, year, month) for year, month in list_of_years_and_months]
    with mp.Pool(mp.cpu_count()) as workers:
        # starmap keeps the order of the tasks, so the output is ordered by year and month
        retrievals = workers.starmap(parse_bsrn_file_with_extra_records, tasks, chunksize=1)

    # remove empty retrievals (files not found or with no supported logical records)
    if not (retrievals := [retr for retr in retrievals if len(retr) > 0]):
        logger.warning(f"no data retrieved for {site=}, {years=}, and {months=}")
        return

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

    if reduced:
        must_variables = ["global_horizontal_avg", "direct_normal_avg", "diffuse_horizontal_avg"]
        data = data[must_variables].rename(columns={var: MAPPING_OF_NAMES[var]["short_name"] if var in MAPPING_OF_NAMES else var
                                           for var in must_variables})

    if extra_output is not None:
        extra_data = {}
        for lr in extra_output:
            this_data = list(filter(None, [clean_data_retrieval(retr, lr=lr) for retr in retrievals]))
            if not this_data:
                logger.warning(f"no data retrieved for logical record {lr} in {site=}, {years=}, and {months=}")
                extra_data[lr] = None
            else:
                extra_data[lr] = pd.concat(this_data, axis=0)

    if centered:
        data = data.set_index(data.index + pd.to_timedelta("30s"))
        if extra_output is not None:
            for lr, df in extra_data.items():
                if df is not None:
                    extra_data[lr] = df.set_index(df.index + pd.to_timedelta("30s"))

    if filled:
        dense_times = pd.date_range(data.index.min(), data.index.max(), freq="1min", inclusive="both", tz="UTC")
        data = data.reindex(dense_times)
        if extra_output is not None:
            for lr, df in extra_data.items():
                if df is not None:
                    dense_times = pd.date_range(df.index.min(), df.index.max(), freq="1min", inclusive="both", tz="UTC")
                    extra_data[lr] = df.reindex(dense_times)

    allsite_metadata = load_metadata().get(site)
    custom_metadata = {key: allsite_metadata[key] for key in allsite_metadata.keys()
                       if key not in ["latitude", "longitude", "altitude"]}
    custom_metadata["timestamp_alignment"] = "center" if centered else "start"

    data = SolarDataFrame(
        data,
        latitude=allsite_metadata["latitude"],
        longitude=allsite_metadata["longitude"],
        elevation=allsite_metadata["altitude"],
        custom_metadata=custom_metadata)

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

    if extra_output is not None:
        return data, metadata, extra_data
    return data, metadata


@dataclass
class LogicalRecord:
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
        path = helpers.fetch_site_data_from_ftp(
            path.name, path.parent, timeout=timeout
        )

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
