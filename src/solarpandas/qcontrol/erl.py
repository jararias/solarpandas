"""Extremely rare limits.

Source: ...
"""

import numpy as np
from loguru import logger

from ..base import SolarDataFrame
from .dtype import QCFlagEnum
from .helpers import construct_flag_series

logger.disable(__name__)
logger = logger.opt(colors=True)


def test_ghi(sdf: SolarDataFrame):
    """Test that GHI is within extremely rare limits.
    Source: ...
    """

    name = "ghi_erl"

    # check that I have what I need: ghi, in this case
    if "ghi" not in sdf.columns:
        logger.warning("`ghi` column not found in dataframe. Test not possible.")
        test_result = np.full(len(sdf), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)
        return construct_flag_series(sdf, name, test_result)

    # compute whatever I need to apply the test
    ghi = sdf["ghi"]
    eth = sdf.solpos.eth
    cosz = sdf.solpos.cosz
    min_value = -2.0  # W m-2, to allow for measurement noise when the sun is just below the horizon
    max_value = 50 + 1.20 * eth * (cosz**1.2)  # W m-2, empirical upper limit

    # compute where the test can be evaluated (verifiable),
    # where it fails (failed) and where it passes (passed)
    verifiable = ghi.notna() & max_value.notna()
    failed = verifiable & (ghi.lt(min_value) | ghi.gt(max_value))
    passed = verifiable & (ghi.ge(min_value) & ghi.le(max_value))

    # construct the flag: -1 for failed, 0 for not verifiable, 1 for passed
    test_result = np.full(len(ghi), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)
    test_result[verifiable & failed] = QCFlagEnum.FAILED.value
    test_result[verifiable & passed] = QCFlagEnum.PASSED.value

    return construct_flag_series(sdf, name, test_result)


def test_dif(sdf: SolarDataFrame):
    """Test that DIF is within extremely rare limits.
    Source: ...
    """

    name = "dif_erl"

    # check that I have what I need: dif, in this case
    if "dif" not in sdf.columns:
        logger.warning("`dif` column not found in dataframe. Test not possible.")
        test_result = np.full(len(sdf), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)
        return construct_flag_series(sdf, name, test_result)

    # compute whatever I need to apply the test
    dif = sdf["dif"]
    eth = sdf.solpos.eth
    cosz = sdf.solpos.cosz
    min_value = -2.0  # W m-2, to allow for measurement noise when the sun is just below the horizon
    max_value = 30 + 0.75 * eth * (cosz**1.2)  # W m-2, empirical upper limit

    # compute where the test can be evaluated (verifiable),
    # where it fails (failed) and where it passes (passed)
    verifiable = dif.notna() & max_value.notna()
    failed = verifiable & (dif.lt(min_value) | dif.gt(max_value))
    passed = verifiable & (dif.ge(min_value) & dif.le(max_value))

    # construct the flag: -1 for failed, 0 for not verifiable, 1 for passed
    test_result = np.full(len(dif), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)
    test_result[verifiable & failed] = QCFlagEnum.FAILED.value
    test_result[verifiable & passed] = QCFlagEnum.PASSED.value

    return construct_flag_series(sdf, name, test_result)


def test_dni(sdf: SolarDataFrame):
    """Test that DNI is within extremely rare limits.
    Source: ...
    """

    name = "dni_erl"

    # check that I have what I need: dni, in this case
    if "dni" not in sdf.columns:
        logger.warning("`dni` column not found in dataframe. Test not possible.")
        test_result = np.full(len(sdf), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)
        return construct_flag_series(sdf, name, test_result)

    # compute whatever I need to apply the test
    dni = sdf["dni"]
    eth = sdf.solpos.eth
    cosz = sdf.solpos.cosz
    min_value = -2.0  # W m-2, to allow for measurement noise when the sun is just below the horizon
    max_value = 10 + 0.95 * eth * (cosz**0.2)  # W m-2, empirical upper limit

    # compute where the test can be evaluated (verifiable),
    # where it fails (failed) and where it passes (passed)
    verifiable = dni.notna() & max_value.notna()
    failed = verifiable & (dni.lt(min_value) | dni.gt(max_value))
    passed = verifiable & (dni.ge(min_value) & dni.le(max_value))

    # construct the flag: -1 for failed, 0 for not verifiable, 1 for passed
    test_result = np.full(len(dni), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)
    test_result[verifiable & failed] = QCFlagEnum.FAILED.value
    test_result[verifiable & passed] = QCFlagEnum.PASSED.value

    return construct_flag_series(sdf, name, test_result)
