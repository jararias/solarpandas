
from functools import lru_cache

import pandas as pd
from loguru import logger

from ..base import SolarDataFrame, SolarSeries
from ..qcontrol import qcrad

logger.disable(__name__)
logger = logger.opt(colors=True)


# The dataframes of pandas are, by design, mutable and unhashable. To be able
# to cache QC results based on the content of the dataframe, we need a hashable
# wrapper that computes a hash based on the content of the dataframe. This is
# what HashableDF does. It computes a hash based on the content of the dataframe
# (including index) and allows us to use it as a key for caching QC results.
class HashableDF:
    def __init__(self, unhashable_df: SolarDataFrame | pd.DataFrame):
        self.dataframe = unhashable_df
        # pd.util.has_pandas_object devuelve un array de hashes para cada fila,
        # sumamos para obtener un hash que representa el contenido de todo el DataFrame
        self._hash = int(pd.util.hash_pandas_object(self.dataframe, index=True).sum())

    def __hash__(self):
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HashableDF):
            return False
        return self.dataframe.equals(other.dataframe)


@lru_cache(maxsize=None)
def _run_cached_qc(hashdf: HashableDF) -> SolarDataFrame:
    """Compute cached quality control."""
    logger.debug("performing cached quality control...")

    sdf = hashdf.dataframe

    qc_results = []
    qc_results.append(qcrad.ghi_ppl(sdf))
    qc_results.append(qcrad.dif_ppl(sdf))
    qc_results.append(qcrad.dni_ppl(sdf))
    qc_results.append(qcrad.ghi_erl(sdf))
    qc_results.append(qcrad.dif_erl(sdf))
    qc_results.append(qcrad.dni_erl(sdf))
    qc_results.append(qcrad.Kn_ppl(sdf))
    qc_results.append(qcrad.Kn_erl(sdf))
    qc_results.append(qcrad.KT_erl(sdf))
    qc_results.append(qcrad.K_erl(sdf))
    qc_results.append(qcrad.K_erl_clear(sdf))
    qc_results.append(qcrad.closure(sdf))

    return pd.concat(qc_results, axis=1)


def clear_qc_cache() -> None:
    """Clear the in-memory quality control cache.

    Call this to free memory or force recomputation on the next access.

    Example::

        import solarpandas as sp
        sp.clear_qc_cache()
    """
    _run_cached_qc.cache_clear()
    logger.debug("qc cache cleared")


def get_qc_cache_info():
    """Get information about the current state of the quality control cache.

    Returns:
        dict: A dictionary containing cache statistics such as hits, misses,
        and current size.

    Example::

        import solarpandas as sp
        info = sp.get_qc_cache_info()
        print(info)
    """
    info = _run_cached_qc.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "current_size": info.currsize,
        "max_size": info.maxsize,
    }


@pd.api.extensions.register_dataframe_accessor("qc")
class QualityControlAccessor:
    """Accessor for computing quality control flags and related results."""

    def __init__(self, sdf_obj):
        self._sdf = self._validate(sdf_obj)
        self._results = _run_cached_qc(HashableDF(self._sdf))

    @staticmethod
    def _validate(obj):
        if not isinstance(obj, SolarDataFrame):
            name = obj.__class__.__name__
            raise AttributeError(f"required a SolarDataFrame instance. Got {name}")
        return obj

    def __getitem__(self, key: str) -> SolarSeries:
        if key not in self._results.columns:
            raise KeyError(f"QC test '{key}' not found in results.")
        return self._results[key]

    def __getattr__(self, name: str) -> SolarSeries:
        if name not in self._results.columns:
            raise AttributeError(f"QC test '{name}' not found in results.")
        return self._results[name]

    @property
    def tests(self) -> SolarDataFrame:
        """Return the columns of the QC results."""
        return self._results
