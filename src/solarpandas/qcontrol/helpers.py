
import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from ..base import SolarDataFrame, SolarSeries

logger.disable(__name__)
logger = logger.opt(colors=True)


def construct_flag_series(
    sdf: SolarDataFrame | SolarSeries,
    name: str, test_result: np.ndarray
) -> SolarSeries: 
    """Helper function to construct a SolarSeries of QC flags from a test result array."""
    return SolarSeries(
        data=test_result,
        index=sdf.index,
        latitude=sdf.latitude,
        longitude=sdf.longitude,
        custom_metadata=copy.deepcopy(sdf.custom_metadata),
        name=name,
        dtype="qcflag",
    )


@dataclass
class QCTest:
    name: str
    _test_func: Callable[[SolarDataFrame], np.ndarray[np.int8]]
    _plot_func: Callable[[SolarDataFrame, np.ndarray[np.int8]], plt.Axes] | None = None

    def __call__(self, sdf: SolarDataFrame) -> SolarSeries:
        return construct_flag_series(sdf, self.name, self._test_func(sdf))

    def plot(self, sdf: SolarDataFrame) -> plt.Axes | None:
        if self._plot_func is None:
            logger.warning(f"No plot function defined for test '{self.name}'")
            return None
        test_result = self._test_func(sdf)
        return self._plot_func(sdf, test_result)
