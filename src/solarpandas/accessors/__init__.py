
# ruff: noqa: F401

import abc
import importlib

from loguru import logger
from pandas.api.extensions import register_dataframe_accessor, register_series_accessor
from pandas.core.series import Callable

logger.disable(__name__)
logger = logger.opt(colors=True)


class BaseLazyAccessor(metaclass=abc.ABCMeta):
    """Base class for lazy accessors that load their real implementation on demand."""

    _namespace: str | None = None
    _on_load: Callable | None = None
    _accessor_cls: type | None = None

    def __init__(self, sdf_obj):
        self._sdf = sdf_obj

    def __init_subclass__(cls, namespace: str, on_load: Callable | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._namespace = namespace
        cls._on_load = on_load

    def _load(self):
        cls = self.__class__
        if cls._accessor_cls is None:
            module_namespace, accessor_class_name = cls._namespace.rsplit(".", 1)
            module = importlib.import_module(module_namespace, package=__package__)
            cls._accessor_cls = getattr(module, accessor_class_name)
            if cls._on_load is not None:
                cls._on_load()
        return cls._accessor_cls(self._sdf)

    def __getattr__(self, name):
        return getattr(self._load(), name)


@register_series_accessor("clearsky")
@register_dataframe_accessor("clearsky")
class ClearskyIrradianceAccessor(
    BaseLazyAccessor,
    namespace=".clearsky.ClearskyIrradianceAccessor",
    on_load=lambda: logger.success("Clear-sky extension loaded")
):
    pass

@register_series_accessor("cda")
@register_dataframe_accessor("cda")
class CDAIrradianceAccessor(
    BaseLazyAccessor,
    namespace=".clearsky.CDAIrradianceAccessor",
    on_load=lambda: logger.success("CDA extension loaded")
):
    pass

@register_series_accessor("lta")
@register_dataframe_accessor("lta")
class LTAIrradianceAccessor(
    BaseLazyAccessor,
    namespace=".clearsky.LTAIrradianceAccessor",
    on_load=lambda: logger.success("LTA extension loaded")
):
    pass

@register_dataframe_accessor("qc")
class QualityControlAccessor(
    BaseLazyAccessor,
    namespace=".qcontrol.QualityControlAccessor",
    on_load=lambda: logger.success("Quality-control extension loaded")
):
    pass

@register_series_accessor("flag")
class QCFlagAccessor(
    BaseLazyAccessor,
    namespace=".qcflag.QCFlagAccessor",
    on_load=lambda: logger.success("QC-flag extension loaded")
):
    pass

@register_series_accessor("skyclass")
@register_dataframe_accessor("skyclass")
class SkyClassAccessor(
    BaseLazyAccessor,
    namespace=".skyclass.SkyClassAccessor",
    on_load=lambda: logger.success("Sky classification extension loaded")
):
    pass

@register_series_accessor("solarplot")
@register_dataframe_accessor("solarplot")
class SolarPlotAccessor(
    BaseLazyAccessor,
    namespace=".solarplot.SolarPlotAccessor",
    on_load=lambda: logger.success("Plotting extension loaded")
):
    pass

@register_series_accessor("solpos")
@register_dataframe_accessor("solpos")
class SolarPositionAccessor(
    BaseLazyAccessor,
    namespace=".solpos.SolarPositionAccessor",
    on_load=lambda: logger.success("Solar-position extension loaded")
):
    pass

@register_series_accessor("param")
@register_dataframe_accessor("param")
class ParameterAccessor(
    BaseLazyAccessor,
    namespace=".param.ParameterAccessor",
    on_load=lambda: logger.success("Parameter extension loaded")
):
    pass

@register_dataframe_accessor("pv")
class PVAccessor(
    BaseLazyAccessor,
    namespace=".pvirrad.PVAccessor",
    on_load=lambda: logger.success("PV extension loaded")
):
    pass