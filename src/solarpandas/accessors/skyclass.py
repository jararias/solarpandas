
import warnings

import numpy as np
import pandas as pd
from loguru import logger

from ..base import SolarSeries, SolarDataFrame
from ..helpers import normalize


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

    def reno_and_hansen(self) -> SolarSeries:  #, full_output: bool = False) -> SolarSeries:
        """Return a clear-sky flag based on the Reno and Hansen (2016) algorithm.

        Return a clear-sky flag: True for clear-sky conditions and False otherwise

        [1] Reno and Hansen (2016) doi: 10.1016/j.renene.2015.12.031
        [2] Gueymard et al. (2019) doi: 10.1016/j.rser.2019.04.027

        Code adapted from pvlib-python.

        Parameters
        ----------
        full_output: boolean
            True to get extra outputs (see below)

        Returns
        -------
        If full_output is False (default) returns a ndarray of bools, which is
        True for clear-sky and False otherwise.
        If full_output is True returns a ndarray of bools (same as before) and a
        dictionary with additional internal parameters (see code below)
        """
        from scipy.linalg import hankel
        from scipy.optimize import minimize_scalar

        # (int) length of sliding time window in minutes. Must be greater than 2
        window_length_minutes = 10

        # (float) threshold value for agreement between mean values of measured and
        # clear-sky values in each interval. Defaults to 75 W/m2. See Eq. 6 in [1]
        mean_diff_thresh = 75.

        # (float) threshold value for agreement between maxima of measured and
        # clear-sky values in each interval. Defaults to 75 W/m2. See Eq. 7 in [1]
        max_diff_thresh = 75.

        # (float) lower limit of line length criterion from Eq. 8 in [1]. Criterion
        # satisfied when lower_line_length_thresh < line length difference <
        # upper_line_length_thresh. Defaults to -5
        lower_line_length_thresh = -5.

        # (float) upper limit of line length criterion from Eq. 8 in [1]. Criterion
        # satisfied when lower_line_length_thresh < line length difference <
        # upper_line_length_thresh. Defaults to -5
        upper_line_length_thresh = 10.

        # (float) threshold value in Hz for the agreement between normalized standard
        # deviations of rate of change in irradiance. See Eqs 9 - 11 in [1]
        # Defaults to 0.005.
        var_diff_thresh = 0.005

        # (float) threshold value for agreement between the largest magnitude of change
        # in successive values, see Eqs. 12 - 14 in [1]. Defaults to 8
        slope_dev_thresh = 8.

        # (int) maximum number of loop iterations to apply a different scaling factor
        # to the clear-sky and redetermine clear samples. Must be greater than 0.
        # Defaults to 100
        max_iterations = 100

        # (float) precision of the maximum relative difference between scaling factor
        # values in successive loop interations. Defaults to 4 decimal positions
        rtol_alpha = 4

        nan_to_num = np.nan_to_num

        sdf = normalize(self._sdf[["ghi"]])
        times = sdf.index
        meas_ghi = sdf.ghi.to_numpy()
        csky_ghi = sdf.clearsky.ghi.to_numpy()
        sza = sdf.solpos.zenith.to_numpy()

        assert len(times) == len(meas_ghi) == len(csky_ghi)

        one_minute = np.timedelta64(1, '60s')
        deltas = np.diff(times.values) / one_minute
        if len(np.unique(deltas)) > 1:
            raise ValueError(
                'the time difference between consecutive time steps must be ',
                'constant throughout the input time series')
        period_minutes = deltas[0]

        n_steps = len(times)
        steps_per_window = int(window_length_minutes / period_minutes)

        H = hankel(
            np.arange(steps_per_window),
            np.arange(steps_per_window - 1, n_steps))

        # statistics on measurements
        meas_mean = np.mean(meas_ghi[H], axis=0)
        meas_max = np.max(meas_ghi[H], axis=0)
        meas_diff = np.diff(meas_ghi[H], n=1, axis=0)
        meas_slope = np.diff(meas_ghi[H], n=1, axis=0) / period_minutes

        meas_slope_nstd = np.full(meas_mean.shape, np.nan)
        universe = np.abs(nan_to_num(meas_mean)) > 1e-5
        meas_slope_nstd[universe] = \
            np.std(meas_slope, axis=0, ddof=1)[universe] / meas_mean[universe]
        # meas_slope_nstd = np.std(meas_slope, axis=0, ddof=1) / meas_mean

        meas_line_length = np.sum(
            np.sqrt(meas_diff**2 + period_minutes**2), axis=0)

        # statistics on clear-sky input
        csky_mean = np.mean(csky_ghi[H], axis=0)
        csky_max = np.max(csky_ghi[H], axis=0)
        csky_diff = np.diff(csky_ghi[H], n=1, axis=0)
        csky_slope = np.diff(csky_ghi[H], n=1, axis=0) / period_minutes

        clearsky_flag = np.full_like(meas_ghi, False, dtype='bool')

        alpha = 1.
        alphas = []
        for _ in range(max_iterations):
            csky_line_length = np.sum(
                np.sqrt(alpha**2 * csky_diff**2 + period_minutes**2), axis=0)

            mean_diff = np.abs(meas_mean - alpha * csky_mean)
            max_diff = np.abs(meas_max - alpha * csky_max)
            line_diff = meas_line_length - csky_line_length
            slope_diff = np.abs(meas_slope - alpha * csky_slope)

            # clear-sky conditions
            cond1 = nan_to_num(mean_diff, nan=np.inf) < mean_diff_thresh
            cond2 = nan_to_num(max_diff, nan=np.inf) < max_diff_thresh
            cond3 = (
                (nan_to_num(line_diff, nan=-np.inf) > lower_line_length_thresh) &
                (nan_to_num(line_diff, nan=np.inf) < upper_line_length_thresh)
            )
            cond4 = nan_to_num(meas_slope_nstd, nan=np.inf) < var_diff_thresh
            max_slope_diff = np.max(slope_diff, axis=0)
            cond5 = nan_to_num(max_slope_diff, nan=np.inf) < slope_dev_thresh
            cond6 = (csky_mean != 0.) & ~np.isnan(csky_mean)
            conds = cond1 & cond2 & cond3 & cond4 & cond5 & cond6

            # update clear sky mask
            clearsky_flag[:] = False
            clearsky_flag[np.unique(H[:, conds])] = True

            # update alpha
            alphas.append(alpha)
            csky_meas = meas_ghi[clearsky_flag]
            csky_csky = csky_ghi[clearsky_flag]

            res = minimize_scalar(
                lambda alpha: np.sqrt(
                    np.mean((csky_meas - alpha * csky_csky)**2)
                )
            )
            alpha = res.x

            if round(alpha, rtol_alpha) == round(alphas[-1], rtol_alpha):
                alphas.append(alpha)
                break

        else:
            if max_iterations > 1:
                warnings.warn(
                    f'failed to converge after {max_iterations} iterations',
                    RuntimeWarning)

        clearsky_flag[sza > 85.] = False

        result = SolarSeries(
            data=clearsky_flag,
            index=times,
            latitude=self._sdf.latitude,
            longitude=self._sdf.longitude,
            elevation=self._sdf.elevation,
            custom_metadata=self._sdf.custom_metadata,
            name="reno_and_hansen",
            dtype=np.int8)

        # keep only the original timestamps
        result = result.reindex(self._sdf.index)

        # restore the original index, in particular, if it was tz-naive
        result.index = self._sdf.index

        return result

        # if full_output is True:
        #     n = len(clearsky_flag)
        #     out_dict = {}
        #     out_dict['mean_diff_flag'] = np.full(n, False, 'bool')
        #     out_dict['mean_diff_flag'][:len(cond1)] = cond1
        #     out_dict['max_diff_flag'] = np.full(n, False, 'bool')
        #     out_dict['max_diff_flag'][:len(cond2)] = cond2
        #     out_dict['line_length_flag'] = np.full(n, False, 'bool')
        #     out_dict['line_length_flag'][:len(cond3)] = cond3
        #     out_dict['slope_nstd_flag'] = np.full(n, False, 'bool')
        #     out_dict['slope_nstd_flag'][:len(cond4)] = cond4
        #     out_dict['slope_max_flag'] = np.full(n, False, 'bool')
        #     out_dict['slope_max_flag'][:len(cond5)] = cond5
        #     out_dict['mean_nan_flag'] = np.full(n, False, 'bool')
        #     out_dict['mean_nan_flag'][:len(cond6)] = cond6
        #     out_dict['mean_diff'] = meas_mean - alpha * csky_mean
        #     out_dict['max_diff'] = meas_max - alpha * csky_max
        #     out_dict['line_length'] = meas_line_length - csky_line_length
        #     out_dict['slope_nstd'] = meas_slope_nstd
        #     out_dict['slope_max'] = np.max(meas_slope - alpha * csky_slope, axis=0)
        #     out_dict['alphas'] = alphas

        #     return clearsky_flag, out_dict

        # return clearsky_flag

