
import functools
import re
from io import StringIO
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from loguru import logger

from . import tables

logger.disable(__name__)
logger = logger.opt(colors=True)


def fortran_pattern_to_colspecs(fortran_pattern: str):

    def safe_split(pattern: str) -> list[str]:
        # split by comma, but ignore commas inside parentheses
        safe_pattern = re.sub(r',(?=[^()]*\))', ';', pattern) # temporarily replace commas inside parentheses with semicolons
        return [element.replace(";", ",") for element in safe_pattern.split(',')]

    def expand_compounded_multiplicity(fortran_pattern: str) -> str:
        def expand(pattern):
            # expand patterns like 3(X,I2) into X,I2,X,I2,X,I2
            if match := re.match(r"^(\d+)[(](.+)[)]$", pattern):
                multiplicity, pat = match.groups()
                return ",".join(pat.split(",") * int(multiplicity))
            return pattern
        return ",".join([expand(ele) for ele in safe_split(fortran_pattern)])

    def expand_single_multiplicity(fortran_pattern: str) -> str:
        def expand(pattern):
            # expand patterns like 3X into X,X,X
            if match := re.match(r"^(\d+)(.+)$", pattern):
                multiplicity, pat = match.groups()
                return ",".join([pat] * int(multiplicity))
            return pattern
        return ",".join([expand(ele) for ele in safe_split(fortran_pattern)])

    expanded_pattern = expand_compounded_multiplicity(fortran_pattern)
    expanded_pattern = expand_single_multiplicity(expanded_pattern)

    colspecs = []
    formatters = []
    current_pos = 0
    for element in expanded_pattern.upper().split(","):
        if not (match := re.match(r"^([AFIX])(\d*(?:\.\d+)?)$", element)):
            raise ValueError(f"Invalid fortran pattern: {element}")
        type_, width = match.groups()
        width = 0 if width == "" else width
        if type_ == "A":
            colspecs.append((current_pos, current_pos + int(width)))
            formatters.append(str.strip)
            current_pos += (0 if width == "" else int(width))
        elif type_ == "F":
            if not (match := re.match(r"^(\d+)\.(\d+)$", width)): # validate F width format
                raise ValueError(f"invalid fortran float number pattern: {element}")
            width, _ = match.groups()
            colspecs.append((current_pos, current_pos + int(width)))
            formatters.append(float)
            current_pos += (0 if width == "" else int(width))
        elif type_ == "I":
            colspecs.append((current_pos, current_pos + int(width)))
            formatters.append(int)
            current_pos += (0 if width == "" else int(width))
        elif type_ == "X":
            current_pos += (0 if width == "" else int(width)) + 1
        else:
            raise ValueError(f"unsupported fortran type: {type_}")
    return colspecs, formatters


def parse(
    txt: str,
    fortran_pattern: str | None = None,
    colspecs: list[tuple[int, int]] | None = None,
    formatter: Callable | list[Callable] | None = None,
    on_error: Literal["raise", "ignore"] = "raise",
    default: Any = "undefined"
) -> list[Any]:

    width = max(len("line being parsed"), len("fortran pattern"), len("colspecs"), len("formatter"))
    logger.debug(f"{'line being parsed':>{width}}: <green>{txt}</green>")

    if fortran_pattern is not None and colspecs is None:
        logger.debug(f"{'fortran pattern':>{width}}: {fortran_pattern}")
        colspecs, formatter_ = fortran_pattern_to_colspecs(fortran_pattern)
        if formatter is None:
            formatter = formatter_
    
    logger.debug(f"{'colspecs':>{width}}: {colspecs}")
    logger.debug(f"{'formatter':>{width}}: {formatter}")

    try:
        values = [txt[start:end+1] for start, end in colspecs]
    except Exception as e:
        msg = f"there was an error parsing line:\n{txt}\nwith colspecs: {colspecs}"
        if on_error == "raise":
            raise ValueError(msg) from e
        values = [default] * len(colspecs)
        logger.debug(f"      values: {values}")
        logger.warning(msg + ". Returning dummy values.")
        return values

    if formatter is not None:
        if isinstance(formatter, Callable):
            formatter = [formatter] * len(colspecs)
        formatted_values = []
        for k, (fmt, value) in enumerate(zip(formatter, values)):
            try:
                formatted_values.append(fmt(value))
            except Exception as e:
                msg = (f"there was an error applying formatter to element {k}-th in line:\n"
                       f"{txt}\nwith colspecs: {colspecs}, formatter {fmt} and value `{value}`")
                if on_error == "raise":
                    raise ValueError(msg) from e
                formatted_values.append(default)
                logger.warning(msg + ". Returning dummy value for this element.")
        values = formatted_values
    logger.debug(f"      values: {values}")
    return values


def parse_logical_record_0001(lines: list[str], **kwargs) -> dict[str, Any]:
    elements = {}
    ilines = iter(lines)

    names = ("station_id", "month", "year", "data_version")
    values = parse(next(ilines), fortran_pattern="2(X,I2),X,I4,X,I2")
    elements.update({name: value for name, value in zip(names, values)})

    # second and following data lines
    elements['quantity_measured'] = []
    for line in ilines:
        values = parse(line, fortran_pattern="8(X,I9)")
        elements['quantity_measured'].extend([tables.TableA3[qty_id] for qty_id in values if qty_id in tables.TableA3])

    return elements


def parse_logical_record_0002(lines: list[str], **kwargs) -> dict[str, Any]:
    elements = {}
    ilines = iter(lines)

    values = parse(next(ilines), fortran_pattern="3(X,I2)")
    elements['scientist_changed_on'] = dict(zip(('day', 'hour', 'minute'), values))

    names = ("scientist_name", "scientist_telephone", "scientist_fax")
    values = parse(next(ilines), fortran_pattern="A38,X,A20,X,A20")
    elements.update({name: value for name, value in zip(names, values)})

    names = ("scientist_tcp/ip", "scientist_email")
    values = parse(next(ilines), fortran_pattern="A15,X,A50")
    elements.update({name: value for name, value in zip(names, values)})

    names = ("scientist_address",)
    values = parse(next(ilines), fortran_pattern="A80")
    elements.update({name: value for name, value in zip(names, values)})

    values = parse(next(ilines), fortran_pattern="3(X,I2)")
    elements['deputy_changed_on'] = dict(zip(('day', 'hour', 'minute'), values))

    names = ("deputy_name", "deputy_telephone", "deputy_fax")
    values = parse(next(ilines), fortran_pattern="A38,X,A20,X,A20")
    elements.update({name: value for name, value in zip(names, values)})

    names = ("deputy_tcp/ip", "deputy_email")
    values = parse(next(ilines), fortran_pattern="A15,X,A50")
    elements.update({name: value for name, value in zip(names, values)})

    names = ("deputy_address",)
    values = parse(next(ilines), fortran_pattern="A80")
    elements.update({name: value for name, value in zip(names, values)})

    return elements


def parse_logical_record_0003(lines: list[str], **kwargs) -> dict[str, Any]:
    return {'message': parse(lines[0], fortran_pattern="A80")[0]}


def parse_logical_record_0004(lines: list[str], **kwargs) -> dict[str, Any]:
    elements = {}
    ilines = iter(lines)

    values = parse(next(ilines), fortran_pattern="3(X,I2)")
    elements['scientist_changed_on'] = dict(zip(('day', 'hour', 'minute'), values))

    values = parse(next(ilines), fortran_pattern="2(X,I2)")
    elements["surface_type"] = tables.TableA4.get(values[0], f"unknown surface type {values[0]}")
    elements["topography_type"] = tables.TableA5.get(values[1], f"unknown topography type {values[1]}")

    elements['station_address'] = parse(next(ilines), fortran_pattern="A80")[0]

    names = ("station_telephone", "station_fax")
    values = parse(next(ilines), fortran_pattern="A20,X,A20")
    elements.update({name: value for name, value in zip(names, values)})

    names = ("station_tcp/ip", "station_email")
    values = parse(next(ilines), fortran_pattern="A15,X,A50")
    elements.update({name: value for name, value in zip(names, values)})

    values = parse(next(ilines), fortran_pattern="2(X,F7.3),X,I4,X,A5")
    elements['latitude'] = values[0] - 90.
    elements['longitude'] = values[1] - 180.
    elements['altitude'] = values[2]
    elements['synop_id'] = values[3]

    values = parse(next(ilines), fortran_pattern="3(X,I2)")
    elements['horizon_changed_on'] = dict(zip(('day', 'hour', 'minute'), values))

    values = [parse(line, fortran_pattern="11(X,I3,X,I2)") for line in ilines]
    values = [e for e in functools.reduce(lambda a, b: a + b, values) if e != -1]
    elements['horizon_azimuth'] = values[0::2]
    elements['horizon_elevation'] = values[1::2]

    return elements


def parse_logical_record_0005(lines: list[str], **kwargs) -> dict[str, Any]:
    elements = {}
    ilines = iter(lines)

    values = parse(next(ilines), fortran_pattern="3(X,I2),X,A1")
    elements["radiosonde_changed_on"] = dict(zip(('day', 'hour', 'minute'), values[:3]))
    elements["radiosonde_operating"] = values[3].casefold() == "y"

    names = ("radiosonde_manufacturer", "radiosonde_location", "radiosonde_distance_km",
             "radiosonde_hUTC_1st_launch", "radiosonde_hUTC_2nd_launch", "radiosonde_hUTC_3rd_launch",
             "radiosonde_hUTC_4th_launch", "radiosonde_id")
    values = parse(next(ilines), fortran_pattern="A30,X,A25,X,I3,4(X,I2),X,A5")
    elements.update({name: value for name, value in zip(names, values)})

    values = parse(next(ilines), fortran_pattern="A80")
    elements["radiosonde_remarks"] = values[0]

    return elements


def parse_logical_record_0006(lines: list[str], **kwargs) -> dict[str, Any]:
    elements = {}
    ilines = iter(lines)

    values = parse(next(ilines), fortran_pattern="3(X,I2),X,A1")
    elements["ozone_changed_on"] = dict(zip(('day', 'hour', 'minute'), values[:3]))
    elements["ozone_operating"] = values[3].casefold() == "y"

    names = ("ozone_manufacturer", "ozone_location", "ozone_distance_km", "ozone_id")
    values = parse(next(ilines), fortran_pattern="A30,X,A25,X,I3,X,I5")
    elements.update({name: value for name, value in zip(names, values)})

    values = parse(next(ilines), fortran_pattern="A80")
    elements["ozone_remarks"] = values[0]

    return elements


def parse_logical_record_0007(lines: list[str], **kwargs) -> dict[str, Any]:
    elements = {}
    ilines = iter(lines)

    values = parse(next(ilines), fortran_pattern="3(X,I2)")
    elements['station_history_changed_on'] = dict(zip(('day', 'hour', 'minute'), values))

    values = parse(next(ilines), fortran_pattern="A80")
    elements['station_history_cloud_amount'] = values[0]

    # method est. cloud base height (with instrument)
    values = parse(next(ilines), fortran_pattern="A80")
    elements['station_history_cloud_base_height'] = values[0]

    # method est. cloud liquid water content
    values = parse(next(ilines), fortran_pattern="A80")
    elements['station_history_cloud_liquid_water_content'] = values[0]

    # method est. cloud aerosol vertical distribution
    values = parse(next(ilines), fortran_pattern="A80")
    elements['station_history_aerosol_vertical_distribution'] = values[0]

    # method est. water vapor press
    values = parse(next(ilines), fortran_pattern="A80")
    elements['station_history_water_vapor_pressure'] = values[0]

    # 6 flags indicating if the SYNOP and/or the corresponding
    # quantities of the expanded programme are measured
    values = parse(next(ilines), fortran_pattern="6(X,A1)")
    elements['station_history_synop_flags'] = [flag.casefold() == "y" for flag in values]

    return elements


def parse_logical_record_0008(lines: list[str], **kwargs) -> dict[str, Any]:
    elements = {}
    ilines = iter(lines)

    elements['instruments'] = []

    while True:
        try:
            line = next(ilines)
        except StopIteration:
            break

        instrument = {}
        values = parse(line, fortran_pattern="3(X,I2)")
        instrument['changed_on'] = dict(zip(('day', 'hour', 'minute'), values))

        names = ("manufacturer", "model", "serial_number", "purchase_date", "wrmc_id")
        values = parse(next(ilines), fortran_pattern="A30,X,A15,X,A18,X,A8,X,I5")
        instrument.update({name: value for name, value in zip(names, values)})

        values = parse(next(ilines), fortran_pattern="A80")
        instrument['remarks'] = values[0]

        names = ("pyrgeometer_body_compensation_code", "pyrgeometer_dome_compensation_code",
                 "wavelength_of_band_1", "bandwidth_of_band_1", "wavelength_of_band_2", "bandwidth_of_band_2",
                 "wavelength_of_band_3", "bandwidth_of_band_3", "max_xx_zenith_angle_direct_degrees",
                 "min_xx_spectral_instrument")
        values = parse(next(ilines), fortran_pattern="2(X,I2),6(X,F7.3),2(X,I2)")
        instrument.update({name: value for name, value in zip(names, values)})

        names = ("location_of_calibration", "person_doing_calibration")
        values = parse(next(ilines), fortran_pattern="A30,X,A40")
        instrument.update({name: value for name, value in zip(names, values)})

        names = ("start_of_calibration_period_of_band_1", "end_of_calibration_period_of_band_1",
                 "number_of_comparisons_of_band_1", "mean_calibration_coefficient_of_band_1",
                 "standard_error_of_calibration_coefficient_of_band_1")
        values = parse(next(ilines), fortran_pattern="2(A8,X),I2,2(X,F12.4)")
        instrument.update({name: value for name, value in zip(names, values)})

        names = ("start_of_calibration_period_of_band_2", "end_of_calibration_period_of_band_2",
                 "number_of_comparisons_of_band_2", "mean_calibration_coefficient_of_band_2",
                 "standard_error_of_calibration_coefficient_of_band_2")
        values = parse(next(ilines), fortran_pattern="2(A8,X),I2,2(X,F12.4)")
        instrument.update({name: value for name, value in zip(names, values)})

        names = ("start_of_calibration_period_of_band_3", "end_of_calibration_period_of_band_3",
                 "number_of_comparisons_of_band_3", "mean_calibration_coefficient_of_band_3",
                 "standard_error_of_calibration_coefficient_of_band_3")
        values = parse(next(ilines), fortran_pattern="2(A8,X),I2,2(X,F12.4)")
        instrument.update({name: value for name, value in zip(names, values)})

        remarks1 =parse(next(ilines), fortran_pattern="A80")[0]
        remarks2 = parse(next(ilines), fortran_pattern="A80")[0]
        instrument['remarks'] = "\n".join([remarks1, remarks2])

        elements['instruments'].append(instrument)

    return elements


def parse_logical_record_0009(lines: list[str], **kwargs) -> dict[str, Any]:
    elements = {}
    ilines = iter(lines)

    elements['quantities'] = []

    while True:
        try:
            line = next(ilines)
        except StopIteration:
            break

        quantity = {}
        values = parse(line, fortran_pattern="3(X,I2),X,I9,X,I5,X,I2")
        quantity['changed_on'] = dict(zip(('day', 'hour', 'minute'), values[:3]))
        quantity['quantity_measured'] = tables.TableA3.get(values[3], f"unknown quantity with id {values[3]}")
        quantity['instrument_id'] = values[4]
        quantity['spectral_band_id'] = values[5]
        elements['quantities'].append(quantity)

    return elements


def parse_logical_record_0100(lines: list[str], **kwargs) -> pd.DataFrame:
    """Parser for the logical record 0100, which contains the basic measurements."""

    def warn(msg: str, day: int | None = None):
        header = ""
        if day is not None:
            header += f"<red>LR0100 @ day {day}</red>"
        if day is not None and "path" in kwargs:
            header += f" <red>in {kwargs['path'].name}</red>"
        logger.warning(f"{header}: {msg}")

    def check_day_consistency(df_day: pd.DataFrame) -> pd.DataFrame:

        day_number = df_day.name
        df = df_day.sort_values("minute")

        # normally, the dat.gz files start every day at minute 0 and end at minute 1439,
        # but in some cases (e.g., cam1008.dat.gz) they start at minute 1 and end at minute
        # which can break the logic if the data is read line by line. Hence, I am swithing
        # to read the data in daily blocks of 1440 records
        if (df.iloc[0]["minute"] == 1) and (df.iloc[-1]["minute"] == 1440) and (len(df) == 1440):
            df["minute"] = df["minute"] - 1  # Ajustamos para que el minuto 1 corresponda a 00:00
            warn("minute values start at 1 and end at 1440. Adjusting to start at 0 and end at 1439. "
                 "This happends at some dat.gz files, e.g., cam1008.dat.gz", day_number)

        # drop records with minute values outside the range [0, 1440)
        if not (legal := df["minute"].between(0, 1440, inclusive="left")).all():
            warn(f"minute values are not between 0 and 1439. Skipping {(~legal).sum()} records with "
                 "invalid minute values.", day_number)
            df = df.loc[legal]

        # add hour[0, 23] and minute[0, 59] columns
        hour, minute = np.divmod(df["minute"], 60)
        df["hour"] = hour
        df["minute"] = minute
        df = df.get(["hour", "minute"] + df.columns.drop(["hour", "minute"]).tolist())

        # drop records with hour values outside the range [0, 24)
        if not (legal := df["hour"].between(0, 24, inclusive="left")).all():
            warn(f"hour values are not between 0 and 23. Skipping {(~legal).sum()} records with "
                 "invalid hour values.", day_number)
            df = df.loc[legal]

        # drop records with minute values outside the range [0, 60)
        if not (legal := df["minute"].between(0, 60, inclusive="left")).all():
            warn(f"minute values are not between 0 and 59. Skipping {(~legal).sum()} records with "
                 "invalid minute values.", day_number)
            df = df.loc[legal]

        return df

    COLUMNS_LINE_1 = (
        'day', 'minute', 'global_horizontal_avg', 'global_horizontal_std',
        'global_horizontal_min', 'global_horizontal_max', 'direct_normal_avg',
        'direct_normal_std', 'direct_normal_min', 'direct_normal_max')

    COLUMNS_LINE_2 = (
        'diffuse_horizontal_avg', 'diffuse_horizontal_std', 'diffuse_horizontal_min',
        'diffuse_horizontal_max', 'downward_longwave_avg', 'downward_longwave_std',
        'downward_longwave_min', 'downward_longwave_max', 'air_temperature',
        'relative_humidity', 'atmospheric_pressure')

    buffer = StringIO("\n".join(lines[::2]))
    colspecs, _ = fortran_pattern_to_colspecs("X,I2,X,I4,2(3X,I4,X,F5.1,X,I4,X,I4)")
    line_1 = (pd.read_fwf(buffer, colspecs=colspecs, header=None, na_values=[-999, -99.9])
              .set_axis(COLUMNS_LINE_1, axis=1))

    buffer = StringIO("\n".join(lines[1::2]))
    colspecs, _ = fortran_pattern_to_colspecs("8X,2(3X,I4,X,F5.1,X,I4,X,I4),4X,2(F5.1,X),I4")
    line_2 = (pd.read_fwf(buffer, colspecs=colspecs, header=None, na_values=[-999, -99.9])
              .set_axis(COLUMNS_LINE_2, axis=1))

    data = pd.concat([line_1, line_2], axis=1)
    data = data.groupby("day").apply(lambda df: check_day_consistency(df)).reset_index("day")
    data = data.loc[data["day"].between(1, 31, inclusive="both")]

    return data
  

def parse_logical_record_0300(lines: list[str], **kwargs) -> pd.DataFrame:
    """Parser for the logical record 0300, which contains the basic measurements."""

    def warn(msg: str, day: int | None = None):
        logger.warning(msg if day is None else f"<red>LR0300 @ day{day}</red>: {msg}")

    def check_day_consistency(df_day: pd.DataFrame) -> pd.DataFrame:

        day_number = df_day.name
        df = df_day.sort_values("minute")

        # normally, the dat.gz files start every day at minute 0 and end at minute 1439,
        # but in some cases (e.g., cam1008.dat.gz) they start at minute 1 and end at minute
        # which can break the logic if the data is read line by line. Hence, I am swithing
        # to read the data in daily blocks of 1440 records
        if (df.iloc[0]["minute"] == 1) and (df.iloc[-1]["minute"] == 1440) and (len(df) == 1440):
            df["minute"] = df["minute"] - 1  # Ajustamos para que el minuto 1 corresponda a 00:00
            warn("minute values start at 1 and end at 1440. Adjusting to start at 0 and end at 1439. "
                 "This happends at some dat.gz files, e.g., cam1008.dat.gz", day_number)

        # drop records with minute values outside the range [0, 1440)
        if not (legal := df["minute"].between(0, 1440, inclusive="left")).all():
            warn(f"minute values are not between 0 and 1439. Skipping {(~legal).sum()} records with "
                 "invalid minute values.", day_number)
            df = df.loc[legal]

        # add hour[0, 23] and minute[0, 59] columns
        hour, minute = np.divmod(df["minute"], 60)
        df["hour"] = hour
        df["minute"] = minute
        df = df.get(["hour", "minute"] + df.columns.drop(["hour", "minute"]).tolist())

        # drop records with hour values outside the range [0, 24)
        if not (legal := df["hour"].between(0, 24, inclusive="left")).all():
            warn(f"hour values are not between 0 and 23. Skipping {(~legal).sum()} records with "
                 "invalid hour values.", day_number)
            df = df.loc[legal]

        # drop records with minute values outside the range [0, 60)
        if not (legal := df["minute"].between(0, 60, inclusive="left")).all():
            warn(f"minute values are not between 0 and 59. Skipping {(~legal).sum()} records with "
                 "invalid minute values.", day_number)
            df = df.loc[legal]

        return df

    COLUMNS = (
        'day', 'minute',
        'upward_shortwave_reflected_avg', 'upward_shortwave_reflected_std',
        'upward_shortwave_reflected_min', "upward_shortwave_reflected_max",
        'upward_longwave_avg', 'upward_longwave_std', "upward_longwave_min",
        'upward_longwave_max', 'net_radiation_avg', "net_radiation_std",
        "net_radiation_min", "net_radiation_max")

    buffer = StringIO("\n".join(lines))
    colspecs, _ = fortran_pattern_to_colspecs("X,I2,X,I4,3(3X,I4,X,F5.1,X,I4,X,I4)")
    data = (pd.read_fwf(buffer, colspecs=colspecs, header=None, na_values=[-999, -99.9])
            .set_axis(COLUMNS, axis=1))

    data = data.groupby("day").apply(lambda df: check_day_consistency(df)).reset_index("day")
    data = data.loc[data["day"].between(1, 31, inclusive="both")]

    return data


def parse_logical_record_0500(lines: list[str], **kwargs) -> pd.DataFrame:
    """Parser for the logical record 0500, which contains the basic measurements."""

    def warn(msg: str, day: int | None = None):
        logger.warning(msg if day is None else f"<red>LR0500 @ day{day}</red>: {msg}")

    def check_day_consistency(df_day: pd.DataFrame) -> pd.DataFrame:

        day_number = df_day.name
        df = df_day.sort_values("minute")

        # normally, the dat.gz files start every day at minute 0 and end at minute 1439,
        # but in some cases (e.g., cam1008.dat.gz) they start at minute 1 and end at minute
        # which can break the logic if the data is read line by line. Hence, I am swithing
        # to read the data in daily blocks of 1440 records
        if (df.iloc[0]["minute"] == 1) and (df.iloc[-1]["minute"] == 1440) and (len(df) == 1440):
            df["minute"] = df["minute"] - 1  # Ajustamos para que el minuto 1 corresponda a 00:00
            warn("minute values start at 1 and end at 1440. Adjusting to start at 0 and end at 1439. "
                 "This happends at some dat.gz files, e.g., cam1008.dat.gz", day_number)

        # drop records with minute values outside the range [0, 1440)
        if not (legal := df["minute"].between(0, 1440, inclusive="left")).all():
            warn(f"minute values are not between 0 and 1439. Skipping {(~legal).sum()} records with "
                 "invalid minute values.", day_number)
            df = df.loc[legal]

        # add hour[0, 23] and minute[0, 59] columns
        hour, minute = np.divmod(df["minute"], 60)
        df["hour"] = hour
        df["minute"] = minute
        df = df.get(["hour", "minute"] + df.columns.drop(["hour", "minute"]).tolist())

        # drop records with hour values outside the range [0, 24)
        if not (legal := df["hour"].between(0, 24, inclusive="left")).all():
            warn(f"hour values are not between 0 and 23. Skipping {(~legal).sum()} records with "
                 "invalid hour values.", day_number)
            df = df.loc[legal]

        # drop records with minute values outside the range [0, 60)
        if not (legal := df["minute"].between(0, 60, inclusive="left")).all():
            warn(f"minute values are not between 0 and 59. Skipping {(~legal).sum()} records with "
                 "invalid minute values.", day_number)
            df = df.loc[legal]

        return df

    COLUMNS_LINE_1 = (
        'day', 'minute',
        "uva_global_avg", "uva_global_std", "uva_global_min", "uva_global_max",
        "uvb_direct_avg", "uvb_direct_std", "uvb_direct_min", "uvb_direct_max")

    COLUMNS_LINE_2 = (
        'uvb_global_avg', 'uvb_global_std', 'uvb_global_min', 'uvb_global_max',
        'uvb_diffuse_avg', 'uvb_diffuse_std', 'uvb_diffuse_min', 'uvb_diffuse_max',
        'uvb_reflected_avg', 'uvb_reflected_std', 'uvb_reflected_min', 'uvb_reflected_max')

    buffer = StringIO("\n".join(lines[::2]))
    colspecs, _ = fortran_pattern_to_colspecs("X,I2,X,I4,8(X,F5.1)")
    line_1 = (pd.read_fwf(buffer, colspecs=colspecs, header=None, na_values=[-999, -99.9])
              .set_axis(COLUMNS_LINE_1, axis=1))

    buffer = StringIO("\n".join(lines[1::2]))
    colspecs, _ = fortran_pattern_to_colspecs("8X,12(X,F5.1)")
    line_2 = (pd.read_fwf(buffer, colspecs=colspecs, header=None, na_values=[-999, -99.9])
              .set_axis(COLUMNS_LINE_2, axis=1))

    data = pd.concat([line_1, line_2], axis=1)
    data = data.groupby("day").apply(lambda df: check_day_consistency(df)).reset_index("day")
    data = data.loc[data["day"].between(1, 31, inclusive="both")]

    return data
