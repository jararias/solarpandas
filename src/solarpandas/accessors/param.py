
"""Accessors to retrieve and manipulate parameter metadata in solar series."""

import pandas as pd
from loguru import logger

from ..base import SolarSeries, SolarDataFrame


logger.disable(__name__)
logger = logger.opt(colors=True)


@pd.api.extensions.register_series_accessor("param")
@pd.api.extensions.register_dataframe_accessor("param")
class ParameterAccessor:
    """Accessor for derived irradiance parameters used in QC workflows.

    Examples
    --------
    >>> sdf.param.KT
    >>> sdf.param.Kn
    """

    def __init__(self, sdf_obj):
        self._sdf = self._validate(sdf_obj)

    @staticmethod
    def _validate(obj):
        if not isinstance(obj, SolarDataFrame):
            name = obj.__class__.__name__
            raise AttributeError(f"required a SolarDataFrame instance. Got {name}")
        return obj

    @property
    def KT(self) -> SolarSeries:
        """Return the clearness index KT derived from ``ghi / eth``.

        Returns
        -------
        SolarSeries
            Daytime-clipped KT values in the range ``[1e-3, 1.35]``.
        """
        if "ghi" not in self._sdf.columns:
            logger.warning("`ghi` column not found in dataframe. Cannot compute KT.")
            return self._sdf.clone(pd.NA).iloc[:, 0].rename("KT")
        eth = self._sdf.solpos.eth
        daytime = self._sdf.solpos.zenith < 87.
        return self._sdf["ghi"].divide(eth).where(daytime, 0.).clip(1e-3, 1.35).rename("KT")

    @property
    def K(self) -> SolarSeries:
        """Return diffuse fraction K derived from ``dif / ghi``.

        Returns
        -------
        SolarSeries
            Daytime-clipped K values in the range ``[1e-3, 1.10]``.
        """
        if "ghi" not in self._sdf.columns:
            logger.warning("`ghi` column not found in dataframe. Cannot compute K.")
            return self._sdf.clone(pd.NA).iloc[:, 0].rename("K")
        if "dif" not in self._sdf.columns:
            logger.warning("`dif` column not found in dataframe. Cannot compute K.")
            return self._sdf.clone(pd.NA).iloc[:, 0].rename("K")
        zenith = self._sdf.solpos.zenith
        daytime = zenith < 87.
        return self._sdf["dif"].divide(self._sdf["ghi"]).where(daytime, 0.).clip(1e-3, 1.10).rename("K")

    @property
    def Kn(self) -> SolarSeries:
        """Return normalized beam index Kn derived from ``dni * cosz / eth``.

        Returns
        -------
        SolarSeries
            Daytime-clipped Kn values in the range ``[1e-3, 1.10]``.
        """
        if "dni" not in self._sdf.columns:
            logger.warning("`dni` column not found in dataframe. Cannot compute Kn.")
            return self._sdf.clone(pd.NA).iloc[:, 0].rename("Kn")
        dir = self._sdf["dni"]*self._sdf.solpos.cosz
        daytime = self._sdf.solpos.zenith < 87.
        return dir.divide(self._sdf.solpos.eth).where(daytime, 0.).clip(1e-3, 1.10).rename("Kn")

    # @property
    # def Kcd(self):
    #     daytime = self._sdf.sp.sza < get_option("max_sza")
    #     ghi_cda = self._sdf.cda.ghi
    #     if isinstance(self._sdf, SynDataFrame):
    #         return self._sdf.apply(lambda x: x.divide(ghi_cda).where(daytime, float("nan"))).clip(
    #             0, 1.3
    #         )
    #     return self._sdf.divide(ghi_cda).where(daytime, float("nan")).clip(0, 1.3)

    # @property
    # def Knd(self):
    #     daytime = self._sdf.sp.sza < get_option("max_sza")
    #     dni_cda = self._sdf.cda.dni
    #     if isinstance(self._sdf, SynDataFrame):
    #         return self._sdf.apply(lambda x: x.divide(dni_cda).where(daytime, float("nan"))).clip(
    #             0, 1.3
    #         )
    #     return self._sdf.divide(dni_cda).where(daytime, float("nan")).clip(0, 1.3)

