import functools
import gzip
import itertools
import json
import multiprocessing as mp
import re
import time
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np
import pandas as pd
import platformdirs
from loguru import logger

from ...config import get_option
from . import helpers, lr_parsers, tables
from .types import LogicalRecordName, Month, Site, Year, validate_type
from .utils import time_interpolation

logger.disable(__name__)
logger = logger.opt(colors=True)


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


def __parse_bsrn_file__(site, year, month):
    retrieval = parse_bsrn_file(
        site=site,
        year=year,
        month=month,
        check_remote_on_missing_file=True,
        logical_records=["LR0004", "LR0100"],
        timeout=30)
    return retrieval | {"year": year, "month": month}


def load_data(
    site: Site,
    years: Sequence[Year] | Year,
    months: Sequence[Month] | Month = range(1, 13),
    # center: bool = False,
    # full_output: bool = False,
) -> dict | tuple[dict, dict, dict]:

    site = validate_type(site, Site)
    years = [validate_type(year, Year) for year in np.asarray(years, dtype=int).reshape(-1)]
    months = [validate_type(month, Month) for month in np.asarray(months, dtype=int).reshape(-1)]
    list_of_years_and_months = sorted(itertools.product(years, months), key=lambda x: (x[0], x[1]))

    logger.info(f"loading data for {len(list_of_years_and_months)} BSRN files...")

    tasks = [(site, year, month) for year, month in list_of_years_and_months]
    with mp.Pool(mp.cpu_count()) as workers:
        # starmap keeps the order of the tasks, so the output is ordered by year and month
        retrievals = workers.starmap(__parse_bsrn_file__, tasks, chunksize=1)

    # remove empty retrievals (files not found or with no supported logical records)
    if not (retrievals := [retr for retr in retrievals if len(retr) > 0]):
        logger.warning(f"no data retrieved for {site=}, {years=}, and {months=}")
        return

    def clean_data_retrieval(retrieval: dict, lr: LogicalRecordName) -> pd.DataFrame:
        year = retrieval.get("year")
        month = retrieval.get("month")
        data = retrieval.get(lr)

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

    # TODO: hacer denso tambien data?
    # TODO: metadata
    # TODO: centrar times?
    # TODO: repetir para LR0300 y LR0500 si estan presentes en los retrievals
    # TODO: remapear los nombres y seleccionar por defecto solo las mas relevantes

    return data

    # check_item('latitude')
    # check_item('longitude')
    # check_item('location')
    # check_item('station')
    # check_item('altitude')
    # check_item('horizon_azimuth')
    # check_item('horizon_elevation')
    # check_item('surface_type')
    # check_item('topography_type')
    # check_item('network')

    # data['values'] = pd.concat([retr['values'] for retr in retrieval])

    # # guess time series resolution...
    # data['resolution'] = guess_time_resolution(data['values']).seconds

    # metadata = copy.deepcopy(data)
    # data = metadata.pop('values')
    # return data, metadata  # output port!!


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

    def parse(self):
        if self.parser is None:
            raise ValueError(
                f"no parser available for logical record with id {self.id}"
            )
        logger.debug(
            f"parsing <blue>{self.name}</blue> with parser <blue>{self.parser.__name__}</blue>"
        )
        return self.parser(self.lines)

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
    site: Site,
    year: Year,
    month: Month,
    check_remote_on_missing_file: bool = True,
    logical_records: LogicalRecordName | list[LogicalRecordName] | None = None,
    timeout: int = 30,
) -> dict:

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

    site = validate_type(site, Site)
    year = validate_type(year, Year)
    month = validate_type(month, Month)

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

    local_path = (
        get_database_path() / "ftp" / site / f"{site}{month:02d}{str(year)[-2:]}.dat.gz"
    )
    if check_remote_on_missing_file and not local_path.exists():
        local_path = helpers.fetch_site_data_from_ftp(
            local_path.name, local_path.parent, timeout=timeout
        )

    if not local_path.exists():
        logger.error(
            f"BSRN data file {local_path.name} not found in {local_path.parent}"
        )
        return {}

    logger.info(f"reading file <blue>{local_path.name}</blue> (@ {local_path.parent})")
    with gzip.open(local_path, "rb") as gz:
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
            if lr in [lr.name for lr in supported_logical_records_in_data]:
                logical_records_to_be_parsed.append(lr)
            else:
                logger.warning(
                    f"the logical record {lr} is not supported or is not found in data. Ignoring it."
                )
        logical_records = [
            lr
            for lr in supported_logical_records_in_data
            if lr.name in logical_records_to_be_parsed
        ]

    logger.debug(
        f"the following logical records will be parsed: {[lr.name for lr in logical_records]}"
    )

    contents = {}
    # allsite_metadata = load_metadata()

    for logical_record in sorted(
        logical_records, key=lambda lr: lr.name
    ):  # sort logical records by their name (LRxxxx)
        if not logical_record.parser:
            logger.warning(
                f"unavailable parser for logical record with id {logical_record.name}"
            )
            continue

        lr_data = logical_record.parse()
        contents[logical_record.name] = lr_data

    return contents

    contents["site"] = site
    contents["year"] = year
    contents["month"] = month
    contents["station"] = allsite_metadata[site]["station"]
    contents["location"] = allsite_metadata[site]["location"]
    contents["latitude"] = float(allsite_metadata[site]["latitude"])
    contents["longitude"] = float(allsite_metadata[site]["longitude"])
    contents["altitude"] = float(allsite_metadata[site]["altitude"])

    contents["horizon_azimuth"] = np.array(
        contents["metadata"].get("horizon_azimuth", None)
    )
    contents["horizon_elevation"] = np.array(
        contents["metadata"].get("horizon_elevation", None)
    )
    contents["surface_type"] = tables.TableA4.get(
        contents["metadata"].get("surface_type", None), None
    )
    contents["topography_type"] = contents["metadata"].get(
        "topography_type", contents["metadata"].get("topograpy_type", None)
    )
    contents["topography_type"] = tables.TableA5.get(
        contents["topography_type"], contents["topography_type"]
    )

    contents["network"] = "BSRN"

    logger.debug(f"Site latitude: {contents['latitude']:+.4f}N")
    logger.debug(f"Site longitude: {contents['longitude']:+.4f}E")
    logger.debug(f"Site altitude: {contents['altitude']:+.1f} m.a.s.l.")

    # READ DATA...

    for lr_id in sorted(lr_specs):
        if lr_id not in parser.data_parser_mapping:
            logger.warning(
                f"unsupported data parser for logical record with id {lr_id}: "
                f"({tables.LogicalRecordDescription.get(lr_id, 'unknown logical record')})"
            )
            continue

        lr_desc = lr_specs[lr_id]
        first_line = lr_desc["first_line"]
        last_line = lr_desc["last_line"]

        # I add an entry for each logical record (lrid) because
        # the data for each logical record has independent times
        lr_parser = parser.data_parser_mapping[lr_id]
        logger.debug(f"parsing logical record {lr_id} for data")
        lr_data = lr_parser(txt_data[first_line : last_line + 1])

        # set a DatetimeIndex with the time information in the logical record
        time_dict = {
            "year": year,
            "month": month,
            "day": lr_data["day"],
            "hour": lr_data["hour"],
            "minute": lr_data["minute"],
        }
        times_utc = pd.to_datetime(pd.DataFrame(time_dict), utc=True)
        lr_data = lr_data.set_index(times_utc).drop(columns=["day", "hour", "minute"])

        # add missing timestamps with NaN values, if necessary
        start = pd.to_datetime(f"{year}-{month:02d}-01")
        dense_times = pd.date_range(
            start, start + pd.offsets.MonthBegin(1), freq="1min", inclusive="left"
        )
        lr_data = lr_data.reindex(dense_times)

        contents[f"LR{lr_id}"] = lr_data

        # ERROR: CAM, 2008-05 (cam0508.dat.gz)
        #    there was a problem parsing logical record 0100
        #    hour greater than 23 at: (day=19, hour=24, minute=0)
        # It is exactly the same problem (and solution) than
        # CAM 2013-09 (see below)
        #
        # ERROR: CAM, 2008-10 (cam1008.dat.gz)
        #    there was a problem parsing logical record 0100
        #    hour greater than 23 at: (day=20, hour=24, minute=0)
        # It is exactly the same problem (and solution) than
        # CAM 2013-09 (see below)
        #
        # ERROR: CAM, 2009-08 (cam0809.dat.gz)
        #    there was a problem parsing logical record 0100
        #    hour greater than 23 at: (day=10, hour=24, minute=0)
        # It is exactly the same problem (and solution) than
        # CAM 2013-09 (see below)
        #
        # ERROR: CAM, 2009-11 (cam1109.dat.gz)
        #    there was a problem parsing logical record 0100
        #    hour greater than 23 at: (day=25, hour=24, minute=0)
        # It is exactly the same problem (and solution) than
        # CAM 2013-09 (see below)
        #
        # ERROR: CAM, 2013-07 (cam0713.dat.gz)
        # There are two errors in the time stamps in record C0100
        # for the 2nd day of the month. First, the 2nd minute is
        # missing:
        #   2    0      0 -99.9 -999 -999      0 -99.9 -999 -999
        #               0 -99.9 -999 -999    346 -99.9 -999 -999\
        #    -99.9 -99.9 -999
        #   2    2      0 -99.9 -999 -999      0 -99.9 -999 -999
        #               0 -99.9 -999 -999    351 -99.9 -999 -999\
        #    -99.9 -99.9 -999
        # I simply solve it by inserting a record of missings:
        #   2    0      0 -99.9 -999 -999      0 -99.9 -999 -999
        #               0 -99.9 -999 -999    346 -99.9 -999 -999\
        #    -99.9 -99.9 -999
        #   2    1   -999 -99.9 -999 -999   -999 -99.9 -999 -999
        #            -999 -99.9 -999 -999   -999 -99.9 -999 -999\
        #    -99.9 -99.9 -999
        #   2    2      0 -99.9 -999 -999      0 -99.9 -999 -999
        #               0 -99.9 -999 -999    351 -99.9 -999 -999\
        #    -99.9 -99.9 -999
        # Second, the 2nd day ends with minute 1440, but the maximum
        # allowed minute is 1439, and the 3rd day starts with minute
        # 1, but the first minute should be 0. That is:
        #   2 1439      0 -99.9 -999 -999      0 -99.9 -999 -999
        #               0 -99.9 -999 -999    336 -99.9 -999 -999\
        #    -99.9 -99.9 -999
        #   2 1440   -999 -99.9 -999 -999   -999 -99.9 -999 -999
        #            -999 -99.9 -999 -999   -999 -99.9 -999 -999\
        #    -99.9 -99.9 -999
        #   3    1      0 -99.9 -999 -999      0 -99.9 -999 -999
        #               0 -99.9 -999 -999    337 -99.9 -999 -999\
        #    -99.9 -99.9 -999
        # I simply solve it moving the record with day=2 and
        # minute=1440 to day=3 and minute=0. That is:
        #   2 1439      0 -99.9 -999 -999      0 -99.9 -999 -999
        #               0 -99.9 -999 -999    336 -99.9 -999 -999\
        #    -99.9 -99.9 -999
        #   3    0   -999 -99.9 -999 -999   -999 -99.9 -999 -999
        #            -999 -99.9 -999 -999   -999 -99.9 -999 -999\
        #    -99.9 -99.9 -999
        #   3    1      0 -99.9 -999 -999      0 -99.9 -999 -999
        #               0 -99.9 -999 -999    337 -99.9 -999 -999\
        #    -99.9 -99.9 -999
        #
        # ERROR: CAM, 2013-09 (cam0913.dat.gz)
        # The 11th day ends with minute 1440, but the maximum
        # allowed is 1439, and the 12th day starts with minute 1,
        # but the first minute should be 0. That is:
        #  11 1439      0 -99.9 -999 -999      0 -99.9 -999 -999
        #               0 -99.9 -999 -999    388 -99.9 -999 -999\
        #    -99.9 -99.9 -999
        #  11 1440   -999 -99.9 -999 -999   -999 -99.9 -999 -999
        #            -999 -99.9 -999 -999   -999 -99.9 -999 -999\
        #    -99.9 -99.9 -999
        #  12    1      0 -99.9 -999 -999      0 -99.9 -999 -999
        #               0 -99.9 -999 -999    388 -99.9 -999 -999\
        #    -99.9 -99.9 -999
        # I simply solve it moving the record with day=11 and
        # minute=1440 to day=12 and minute=0. That is:
        #  11 1439      0 -99.9 -999 -999      0 -99.9 -999 -999
        #               0 -99.9 -999 -999    388 -99.9 -999 -999\
        #    -99.9 -99.9 -999
        #  12    0   -999 -99.9 -999 -999   -999 -99.9 -999 -999
        #            -999 -99.9 -999 -999   -999 -99.9 -999 -999\
        #    -99.9 -99.9 -999
        #  12    1      0 -99.9 -999 -999      0 -99.9 -999 -999
        #               0 -99.9 -999 -999    388 -99.9 -999 -999\
        #    -99.9 -99.9 -999
        #
        # ERROR: SMS, 2016-04 (sms0416.dat.gz)
        # From day 24 all days have an extra record "1440". That is:
        #  24 1440   -999 -99.9 -999 -999   -999 -99.9 -999 -999
        #            -999 -99.9 -999 -999   -999 -99.9 -999 -999\
        #    -99.9 -99.9 -999
        # I simply removed them from the file
        #
        # ERROR: SMS, 2016-05 (sms0516.dat.gz)
        # As previous error, but for day 1. I simply removed the
        # record from the file
        #
        # ERROR: SMS, 2016-10 (sms1016.dat.gz)
        # As previous error, but for days 3 and 4. I simply removed
        # the records from the file
        #
        # ERROR: SMS, 2016-11 (sms1116.dat.gz)
        # As previous error, but for day 9. I simply removed the
        # record from the file
        #
        # and many more...

        # # The logical record 0100 is essential because it holds the
        # # solar radiation data. Errors parsing it are not allowed!!
        # if lr_id in ("0100",):
        #     logger.error("there was a problem parsing logical record 0100")
        #     if max(hour) > 23:
        #         msg = "hour greater than 23 at: "
        #         msg += ", ".join(
        #             [f"(year={year}, month={month}, day={day[k]}, hour={hour[k]}, minute={minute[k]})"
        #                 for k in np.argwhere(hour > 23)[0]])
        #         logger.error(msg)
        #     raise exc
        # logger.warning(f"there was a problem parsing logical record {lr_id}. Skipping")

    # for lr_id in sorted(lr_specs):
    #     if ((lr_id not in parser.metadata_parser_mapping) and (lr_id not in parser.data_parser_mapping)):
    #         logger.debug(f"missing parser for logical record {lr_id}")

    # from IPython import embed; embed()

    # # get rid of "stats" columns in logical record 0100 and pile up the
    # # rest in a DataFrame structure
    # if "0100" in contents["logical_records"]:

    #     logger.debug("parsing data in logical record 0100")

    #     def translate(varname):
    #         mapping = {
    #             "global_horizontal": "ghi",
    #             "direct_normal": "dni",
    #             "diffuse_horizontal": "dif",
    #             "downward_longwave": "dlw",
    #             "air_temperature": "tair",
    #             "relative_humidity": "rh",
    #             "atmospheric_pressure": "pressure"
    #         }
    #         for long_name, short_name in mapping.items():
    #             if long_name in varname:
    #                 return varname.replace(long_name, short_name)
    #         return varname

    #     series = {}
    #     times_utc = contents["logical_records"]["0100"]["utc_times"]

    #     for variable in contents["logical_records"]["0100"]:
    #         if variable.endswith("_min") or variable.endswith("_max"):
    #             continue

    #         if variable in ("utc_times", "description"):
    #             continue

    #         var_repr = repr(contents["logical_records"]["0100"][variable])
    #         logger.debug(f'  - getting variable `{variable}`: {var_repr}')
    #         series[translate(variable)] = pd.Series(
    #             data=contents["logical_records"]["0100"][variable],
    #             index=times_utc)

    #     if not series:
    #         logger.debug("<red>empty logical record 0100</red>")
    #         return None

    #     try:
    #         contents["values"] = pd.concat(series, axis=1)
    #     except Exception as exc:  # pylint: disable=broad-except
    #         logger.debug(f"an exception has occurred while retrieving the logical record 0100: {exc}")
    #         return None

    elapsed_time = time.time() - init_time
    logger.debug(f"total elapsed time: {elapsed_time} seconds")

    metadata = contents.pop("metadata")
    logical_records = contents.pop("logical_records")

    #####################################################################
    # THE TIMESTAMP OF THE DATAPOINT REPRESENTS THE STARTING POINT OF   #
    # THE 1-MIN AVERAGE (page 1493, Driemel et al., 2018,               #
    # doi: www.earth-syst-sci-data.net/10/1491/2018/                    #
    #####################################################################
    contents["timestamp_reference"] = "start"

    if center is True:
        contents["values"].index = contents["values"].index + pd.Timedelta(
            30, "seconds"
        )

        # interpolation to be sure that the time grid is dense
        t_s = contents["values"].index[0]
        t_e = contents["values"].index[-1]
        times_1min = pd.date_range(
            f"{t_s.year}-{t_s.month:02d}-01T00:00:30",
            f"{t_e.year}-{t_e.month:02d}-{t_e.daysinmonth}T23:59:30",
            freq=pd.Timedelta(60, "seconds"),
        )
        contents["values"] = time_interpolation(contents["values"], times_1min)

        contents["timestamp_reference"] = "center"

    if full_output is True:
        return contents, metadata, logical_records
    return contents
