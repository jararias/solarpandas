
import numpy as np
import pandas as pd
from loguru import logger

from ..base import SolarSeries, SolarDataFrame


logger.disable(__name__)
logger = logger.opt(colors=True)


@pd.api.extensions.register_series_accessor("skyclass")
@pd.api.extensions.register_dataframe_accessor("skyclass")
class SkyClassAccessor:
    """Accessor for derived irradiance parameters used in sky classification workflows.

    Examples
    --------
    >>> sdf.skyclass.caelus
    """

    def __init__(self, sdf_obj):
        self._sdf = self._validate(sdf_obj)

    @staticmethod
    def _validate(obj):
        if not isinstance(obj, SolarDataFrame):
            name = obj.__class__.__name__
            raise AttributeError(f"required a SolarDataFrame instance. Got {name}")
        return obj

    def caelus(self, **kwargs) -> SolarSeries:
        """Return the caelus sky class.

        Returns
        -------
        SolarSeries
            Caelus sky class labels as integers in the range ``[0, 15]``.
        """
        from caelus.classifier import _classify_from_ensured_dataframe
        from caelus.options import MAX_SZA

        if "ghi" not in self._sdf.columns:
            logger.warning("`ghi` column not found in dataframe. Cannot compute Caelus sky class.")
            return self._sdf.replace_data(1).iloc[:, 0].astype(np.int8).rename("sky_type")

        sdf = self._sdf[["ghi"]].assign(
            sza=self._sdf.solpos.zenith,
            daytime=self._sdf.solpos.zenith < MAX_SZA,
            cosz=self._sdf.solpos.cosz,
            tst=self._sdf.solpos.true_solar_time,
            ghicda=self._sdf.cda.ghi,
            ghics=self._sdf.clearsky.ghi)

        default_kwargs = {}
        default_kwargs.setdefault("engine", "polars")
        default_kwargs.setdefault("apply_filters", True)
        default_kwargs.setdefault("categorical", False)
        default_kwargs.setdefault("full_output", False)
        result = _classify_from_ensured_dataframe(sdf, **(default_kwargs | (kwargs or {})))

        # keep only the original timestamps
        result = result.reindex(self._sdf.index)

        # restore the original index, in particular, if it was tz-naive
        result.index = self._sdf.index

        return SolarSeries(
            data=result.values,
            index=result.index,
            latitude=self._sdf.latitude,
            longitude=self._sdf.longitude,
            elevation=self._sdf.elevation,
            custom_metadata=self._sdf.custom_metadata,
            name=result.name)


