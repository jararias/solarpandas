"""Physically-possible limits.

Source: ...
"""

import copy

import numpy as np
from loguru import logger

from ..base import SolarDataFrame, SolarSeries
from .flag import FlagDtype, QCFlagEnum

logger.disable(__name__)
logger = logger.opt(colors=True)


def _construct_flag_series(sdf: SolarDataFrame | SolarSeries, name: str, test_result: np.ndarray) -> SolarSeries: 
    """Helper function to construct a SolarSeries of QC flags from a test result array."""
    return SolarSeries(
        data=test_result,
        index=sdf.index,
        latitude=sdf.latitude,
        longitude=sdf.longitude,
        custom_metadata=copy.deepcopy(sdf.custom_metadata),
        name=name,
        dtype=FlagDtype(),
    )

def test_ghi(sdf: SolarDataFrame):
    """Test that GHI is within physically-possible limits."""

    name = "ghi_ppl"

    # check that I have what I need: ghi, in this case
    if "ghi" not in sdf.columns:
        logger.warning("`ghi` column not found in dataframe. Test not possible.")
        test_result = np.full(len(sdf), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)
        return _construct_flag_series(sdf, name, test_result)

    # compute whatever I need to apply the test
    ghi = sdf["ghi"]
    eth = sdf.solpos.eth
    cosz = sdf.solpos.cosz
    max_value = 100 + 1.50 * eth * (cosz**1.2)

    # compute where the test can be evaluated (verifiable),
    # where it fails (failed) and where it passes (passed)
    verifiable = ghi.notna()
    failed = ghi.lt(-4.0) | ghi.gt(max_value)
    passed = ghi.ge(-4.0) & ghi.le(max_value)

    # construct the flag: -1 for failed, 0 for not verifiable, 1 for passed
    test_result = np.full(len(ghi), QCFlagEnum.NOT_VERIFIABLE.value, dtype=np.int8)
    test_result[verifiable & failed] = QCFlagEnum.FAILED.value
    test_result[verifiable & passed] = QCFlagEnum.PASSED.value

    return _construct_flag_series(sdf, name, test_result)
