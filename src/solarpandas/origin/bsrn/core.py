
import copy
import gzip
import itertools
import json
import multiprocessing as mp
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import platformdirs
from loguru import logger

from ...config import get_option
from . import helpers, parser, tables
from .types import Month, Site, Year, validate_type
from .utils import time_interpolation, guess_time_resolution

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


def load_metadata(update: Literal["auto"] | bool = "auto"):

    def get_file_age(path: Path):
        if not path.exists():
            return np.inf
        datetime_created = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        file_age = datetime.now(timezone.utc) - datetime_created
        return file_age.total_seconds() / (24 * 3600)  # seconds to days

    metadata_path = get_database_path() / "ftp" / "metadata.json"
    file_age_days = get_file_age(metadata_path)

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
    months: Sequence[Month] | Month = range(1, 13),
    center: bool = False,
    full_output: bool = False,
    check_remote_on_missing_file: bool = True,
    timeout: int = 30,
) -> dict | tuple[dict, dict, dict]:

    site = validate_type(site, Site)

    if not isinstance(years, (int, float)) and not isinstance(months, (int, float)):

        years = [validate_type(year, Year) for year in np.asarray(years, dtype=int).reshape(-1)]
        months = [validate_type(month, Month) for month in np.asarray(months, dtype=int).reshape(-1)]
        list_of_years_and_months = sorted(itertools.product(years, months), key=lambda x: (x[0], x[1]))

        if full_output:
            logger.warning("full_output is only supported for single year/month combinations. Ignoring it.")

        tasks = []
        for year, month in list_of_years_and_months:
            tasks.append((site, year, month, center, False, check_remote_on_missing_file, timeout))

        workers = mp.Pool(mp.cpu_count())
        run = workers.starmap_async(load_data, tasks, chunksize=1)
        while not run.ready():
            pass

        workers.close()
        workers.join()
        retrieval = run.get()  # potentially unordered data!!

        # remove erroneous or missing retrievals
        if len(list_of_years_and_months) > 1:
            logger.info("appending monthly files...")

        retrieval = [retr for retr in filter(None, retrieval) if 'values' in retr]
        if not retrieval:
            logger.warning('missing data')
            return

        # sort the retrievals chronologically
        retrieval = sorted(retrieval, key=lambda retr: (retr['year'], retr['month']))

        # and concatenate them...
        data = retrieval[0]
        data.pop('year')
        data.pop('month')

        def check_item(item):
            value = data.setdefault(item, None)
            if value is None:
                # if value is None, search all the retrievals and
                # take the first value that is not None
                values = list(filter(None, [retr.get(item, None) for retr in retrieval]))  # use filter to remove None's
                if values:
                    data[item] = values[0]

        check_item('latitude')
        check_item('longitude')
        check_item('location')
        check_item('station')
        check_item('altitude')
        check_item('horizon_azimuth')
        check_item('horizon_elevation')
        check_item('surface_type')
        check_item('topography_type')
        check_item('network')

        data['values'] = pd.concat([retr['values'] for retr in retrieval])

        # guess time series resolution...
        data['resolution'] = guess_time_resolution(data['values']).seconds

        metadata = copy.deepcopy(data)
        data = metadata.pop('values')
        return data, metadata  # output port!!

    else:
        return load_bsrn_file(
            site=site,
            year=validate_type(years, Year),
            month=validate_type(months, Month),
            center=center,
            full_output=full_output,
            check_remote_on_missing_file=check_remote_on_missing_file,
            timeout=timeout)


def load_bsrn_file(
    site: Site,
    year: Year,
    month: Month,
    center: bool = False,
    full_output: bool = False,
    check_remote_on_missing_file: bool = True,
    timeout: int = 30,
) -> dict | tuple[dict, dict, dict]:


    site = validate_type(site, Site)
    year = validate_type(year, Year)
    month = validate_type(month, Month)

    local_path = get_database_path() / "ftp" / site / f"{site}{month:02d}{str(year)[-2:]}.dat.gz"
    if check_remote_on_missing_file and not local_path.exists():
        local_path = helpers.fetch_site_data_from_ftp(
            local_path.name, local_path.parent, timeout=timeout)

    if not local_path.exists():
        raise FileNotFoundError(f"BSRN data file {local_path.name} not found in {local_path.parent}")

    logger.info(f"reading file {local_path.name} (@ {local_path.parent})")
    with gzip.open(local_path, "rb") as gz:
        txt_data = [line.rstrip().decode("utf-8") for line in gz.readlines()]

    # cambio `logical_records` por `lr_specs`
    lr_specs = parser.find_logical_record_bounds(txt_data)

    contents = {}
    init_time = time.time()

    # READ METADATA... (must be read before data records)

    contents["metadata"] = {}
    allsite_metadata = load_metadata()

    for lr_id in sorted(lr_specs):
        if lr_id not in parser.metadata_parser_mapping:
            logger.warning(f"unavailable metadata parser for logical record with id {lr_id}")
            continue

        lr_desc = lr_specs[lr_id]
        first_line = lr_desc["first_line"]
        last_line = lr_desc["last_line"]

        lr_contents = {}

        lr_parser = parser.metadata_parser_mapping[lr_id]
        logger.debug(f"parsing logical record {lr_id} for metadata")
        lr_contents = lr_parser(txt_data[first_line : last_line + 1])
        for attr_name in lr_contents:
            logger.debug(f"  - attribute `{attr_name}` retrieved")
        contents["metadata"].update(lr_contents)

    contents["site"] = site
    contents["year"] = year
    contents["month"] = month
    contents["station"] = allsite_metadata[site]["station"]
    contents["location"] = allsite_metadata[site]["location"]
    contents["latitude"] = float(allsite_metadata[site]["latitude"])
    contents["longitude"] = float(allsite_metadata[site]["longitude"])
    contents["altitude"] = float(allsite_metadata[site]["altitude"])

    contents["horizon_azimuth"] = (
        np.array(contents["metadata"].get("horizon_azimuth", None)))
    contents["horizon_elevation"] = (
        np.array(contents["metadata"].get("horizon_elevation", None)))
    contents["surface_type"] = tables.TableA4.get(
        contents["metadata"].get("surface_type", None), None)
    contents["topography_type"] = contents["metadata"].get(
        "topography_type",
        contents["metadata"].get("topograpy_type", None))
    contents["topography_type"] = tables.TableA5.get(
        contents["topography_type"], contents["topography_type"])

    contents["network"] = "BSRN"

    logger.debug(f"Site latitude: {contents["latitude"]:+.4f}N")
    logger.debug(f"Site longitude: {contents["longitude"]:+.4f}E")
    logger.debug(f"Site altitude: {contents["altitude"]:+.1f} m.a.s.l.")

    # READ DATA...

    contents["logical_records"] = {}

    for lr_id in sorted(lr_specs):

        if lr_id not in parser.data_parser_mapping:
            if int(lr_id) > 99:
                descr = tables.LogicalRecordDescription.get(lr_id, None)
                msg = (f"Unavailable data parser for logical record with id {lr_id}")
                if descr is not None:
                    msg += f": {descr}"
                logger.warning(msg)
            continue

        try:
            lr_desc = lr_specs[lr_id]
            first_line = lr_desc["first_line"]
            last_line = lr_desc["last_line"]

            # I add an entry for each logical record (lrid) because
            # the data for each logical record has independent times
            lr_parser = parser.data_parser_mapping[lr_id]
            logger.debug(f"parsing logical record {lr_id} for data")
            lr_contents = lr_parser(txt_data[first_line:last_line + 1])

            lr_contents = {key: value for key, value in lr_contents.items() if not np.all(np.isnan(value))}
            lr_contents["description"] = tables.LogicalRecordDescription.get(lr_id, "unavailable")

            day = lr_contents.pop("day")
            hour = lr_contents.pop("hour")
            minute = lr_contents.pop("minute")
            kwargs = dict(second=0, microsecond=0, tzinfo=None)
            utc_times = [datetime(year, month, day[k], hour[k], minute[k], **kwargs)
                         for k in range(len(day))]
            lr_contents["utc_times"] = np.array(utc_times)

            for attr_name in lr_contents:
                logger.debug(f"  - attribute `{attr_name}` retrieved")

            contents["logical_records"][lr_id] = {}
            contents["logical_records"][lr_id].update(lr_contents)

        except ValueError as exc:
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

            # The logical record 0100 is essential because it holds the
            # solar radiation data. Errors parsing it are not allowed!!
            if lr_id in ("0100",):
                logger.error("there was a problem parsing logical record 0100")
                if max(hour) > 23:
                    msg = "hour greater than 23 at: "
                    msg += ", ".join(
                        [f"(year={year}, month={month}, day={day[k]}, hour={hour[k]}, minute={minute[k]})"
                         for k in np.argwhere(hour > 23)[0]])
                    logger.error(msg)
                raise exc
            logger.warning(f"there was a problem parsing logical record {lr_id}. Skipping")

    for lr_id in sorted(lr_specs):
        if ((lr_id not in parser.metadata_parser_mapping) and (lr_id not in parser.data_parser_mapping)):
            logger.debug(f"missing parser for logical record {lr_id}")

    # from IPython import embed; embed()

    # get rid of "stats" columns in logical record 0100 and pile up the
    # rest in a DataFrame structure
    if "0100" in contents["logical_records"]:

        logger.debug("parsing data in logical record 0100")

        def translate(varname):
            mapping = {
                "global_horizontal": "ghi",
                "direct_normal": "dni",
                "diffuse_horizontal": "dif",
                "downward_longwave": "dlw",
                "air_temperature": "tair",
                "relative_humidity": "rh",
                "atmospheric_pressure": "pressure"
            }
            for long_name, short_name in mapping.items():
                if long_name in varname:
                    return varname.replace(long_name, short_name)
            return varname

        series = {}
        times_utc = contents["logical_records"]["0100"]["utc_times"]

        for variable in contents["logical_records"]["0100"]:
            if variable.endswith("_min") or variable.endswith("_max"):
                continue

            if variable in ("utc_times", "description"):
                continue

            var_repr = repr(contents["logical_records"]["0100"][variable])
            logger.debug(f'  - getting variable `{variable}`: {var_repr}')
            series[translate(variable)] = pd.Series(
                data=contents["logical_records"]["0100"][variable],
                index=times_utc)

        if not series:
            logger.debug("<red>empty logical record 0100</red>")
            return None

        try:
            contents["values"] = pd.concat(series, axis=1)
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug(f"an exception has occurred while retrieving the logical record 0100: {exc}")
            return None

    elapsed_time = time.time() - init_time
    logger.debug(f"total elapsed time: {elapsed_time} seconds")

    if "values" not in contents:
        return None

    metadata = contents.pop("metadata")
    logical_records = contents.pop("logical_records")

    with np.printoptions(threshold=5):
        logger.debug("Retrieved data:")
        for k, v in contents.items():
            if k == "values":
                logger.debug(f"  - {k}: \n{repr(v)}")
                continue
            logger.debug(f"  - {k}: {repr(v)}")

    #####################################################################
    # THE TIMESTAMP OF THE DATAPOINT REPRESENTS THE STARTING POINT OF   #
    # THE 1-MIN AVERAGE (page 1493, Driemel et al., 2018,               #
    # doi: www.earth-syst-sci-data.net/10/1491/2018/                    #
    #####################################################################
    contents['timestamp_reference'] = 'start'

    if center is True:

        contents['values'].index = contents['values'].index + pd.Timedelta(30, 'seconds')

        # interpolation to be sure that the time grid is dense
        t_s = contents['values'].index[0]
        t_e = contents['values'].index[-1]
        times_1min = pd.date_range(
            f'{t_s.year}-{t_s.month:02d}-01T00:00:30',
            f'{t_e.year}-{t_e.month:02d}-{t_e.daysinmonth}T23:59:30',
            freq=pd.Timedelta(60, 'seconds'))
        contents['values'] = time_interpolation(contents['values'], times_1min)

        contents['timestamp_reference'] = 'center'

    if full_output is True:
        return contents, metadata, logical_records
    return contents
