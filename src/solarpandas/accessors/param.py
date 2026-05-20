

# @pd.api.extensions.register_series_accessor("param")
# @pd.api.extensions.register_dataframe_accessor("param")
# class ParameterAccessor:
#     def __init__(self, sdf_obj):
#         self._validate(sdf_obj)
#         self._sdf = sdf_obj

#     @staticmethod
#     def _validate(obj):
#         if not isinstance(obj, (SynSeries, SynDataFrame)):
#             raise AttributeError("required a SynSeries or SynDataFrame instance")

#     @property
#     def Kcd(self):
#         daytime = self._sdf.sp.sza < get_option("max_sza")
#         ghi_cda = self._sdf.cda.ghi
#         if isinstance(self._sdf, SynDataFrame):
#             return self._sdf.apply(lambda x: x.divide(ghi_cda).where(daytime, float("nan"))).clip(
#                 0, 1.3
#             )
#         return self._sdf.divide(ghi_cda).where(daytime, float("nan")).clip(0, 1.3)

#     @property
#     def Knd(self):
#         daytime = self._sdf.sp.sza < get_option("max_sza")
#         dni_cda = self._sdf.cda.dni
#         if isinstance(self._sdf, SynDataFrame):
#             return self._sdf.apply(lambda x: x.divide(dni_cda).where(daytime, float("nan"))).clip(
#                 0, 1.3
#             )
#         return self._sdf.divide(dni_cda).where(daytime, float("nan")).clip(0, 1.3)

#     @property
#     def KT(self):
#         daytime = self._sdf.sp.sza < get_option("max_sza")
#         eth = self._sdf.sp.eth
#         if isinstance(self._sdf, SynDataFrame):
#             return self._sdf.apply(lambda x: x.divide(eth).where(daytime, float("nan"))).clip(
#                 0, 1.3
#             )
#         return self._sdf.divide(eth).where(daytime, float("nan")).clip(0, 1.3)

