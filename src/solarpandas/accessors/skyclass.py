
import warnings
from functools import wraps
from typing import Literal

import numpy as np
from loguru import logger

from ..base import SolarSeries, SolarDataFrame
from ..helpers import normalize

logger.disable(__name__)
logger = logger.opt(colors=True)


def suppress_warnings(category):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=category)
                return func(*args, **kwargs)
        return wrapper
    return decorator


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

    def caelus(
        self,
        engine: Literal["pandas", "polars"] = "polars",
        apply_filters: bool = True,
        categorical: bool = False,
        full_output: bool = False
    ) -> SolarSeries:
        """Return the caelus sky class.

        Returns
        -------
        SolarSeries
            Caelus sky class labels as integers in the range ``[0, 15]``.
        """
        from caelus.classifier import _classify_from_ensured_dataframe
        from caelus.options import MAX_SZA

        if "ghi" not in self._sdf.columns:
            logger.warning("`ghi` column not found in dataframe. Cannot "
                           "compute Caelus 6-class sky type.")
            return self._sdf.replace_data(1).iloc[:, 0].astype(np.int8).rename("caelus")

        sdf = self._sdf[["ghi"]].assign(
            sza=self._sdf.solpos.zenith,
            daytime=self._sdf.solpos.zenith < MAX_SZA,
            cosz=self._sdf.solpos.cosz,
            tst=self._sdf.solpos.true_solar_time,
            ghicda=self._sdf.cda.ghi,
            ghics=self._sdf.clearsky.ghi)

        result = _classify_from_ensured_dataframe(
            sdf,
            engine=engine,
            apply_filters=apply_filters,
            categorical=categorical,
            full_output=full_output)

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
            name="caelus")

    def reno_and_hansen(
        self,
        mean_diff_thresh: float = 75.,
        max_diff_thresh: float = 75.,
        lower_line_length_thresh: float = -5.,
        upper_line_length_thresh: float = 10.,
        var_diff_thresh: float = 0.005,
        slope_dev_thresh: float = 8.,
        max_iterations: int = 100,
        rtol_alpha: float = 4,
    ) -> SolarSeries:
        """Return a clear-sky flag based on the Reno and Hansen (2016) algorithm.

        Return a clear-sky flag: True for clear-sky conditions and False otherwise

        [1] Reno and Hansen (2016) doi: 10.1016/j.renene.2015.12.031
        [2] Gueymard et al. (2019) doi: 10.1016/j.rser.2019.04.027

        Code adapted from pvlib-python.

        Parameters
        ----------
        mean_diff_thresh: float
            Threshold value for agreement between mean values of measured and
            clear-sky values in each interval. Defaults to 75 W/m2. See
            Eq. 6 in [1]
        max_diff_thresh: float
            Threshold value for agreement between maxima of measured and
            clear-sky values in each interval. Defaults to 75 W/m2. See
            Eq. 7 in [1]
        lower_line_length_thresh: float
            Lower limit of line length criterion from Eq. 8 in [1]. Criterion
            satisfied when lower_line_length_thresh < line length difference <
            upper_line_length_thresh. Defaults to -5
        upper_line_length_thresh: float
            Upper limit of line length criterion from Eq. 8 in [1]. Criterion
            satisfied when lower_line_length_thresh < line length difference <
            upper_line_length_thresh. Defaults to -5
        var_diff_thresh: float
            Threshold value in Hz for the agreement between normalized standard
            deviations of rate of change in irradiance. See Eqs 9 - 11 in [1]
            Defaults to 0.005.
        slope_dev_thresh: float
            Threshod value for agreement between the largest magnitude of change
            in successive values, see Eqs. 12 - 14 in [1]. Defaults to 8
        max_iterations: integer
            Maximum number of loop iterations to apply a different scaling factor
            to the clear-sky and redetermine clear samples. Must be greater than
            0. Defaults to 100
        rtol_alpha: float
            Precision of the maximum relative difference between scaling factor
            values in successive loop interations. Defaults to 4 decimal positions

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

        nan_to_num = np.nan_to_num

        if "ghi" not in self._sdf.columns:
            logger.warning("`ghi` column not found in dataframe. Cannot compute "
                           "Reno and Hansen 2-class sky type.")
            return self._sdf.replace_data(1).iloc[:, 0].astype(np.int8).rename("reno_and_hansen")

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
            dtype=np.int8)

        # keep only the original timestamps
        result = result.reindex(self._sdf.index)

        # restore the original index, in particular, if it was tz-naive
        result.index = self._sdf.index
        result.name = "reno_and_hansen"

        return result

    def bright_and_sun(
        self,
        csd: Literal["sunpath", "allsky"] = 'allsky',
        # full_output=False,
        window_length_minutes: int = 10,
        max_iterations: int = 20,
        rtol_alpha: float = 4
    ) -> SolarSeries:
        """
        Cloud screening algorithm proposed by Bright et al. (2020) for 1-min
        irradiance data. Requires GHI and DIF, plus GHI and DIF clear-sky values

        Return a clear-sky flag: True for cloudless conditions and False otherwise.

        [1] Bright et al. (2020) doi: 10.1016/j.rser.2020.109706

        Code adapted from the Matlab code available in
        https://github.com/JamieMBright/csd-library (models/BrightSun2020CSDc.m)

        Parameters
        ----------
        csd : `allsky` or `sunpath`
            `allsky` to detect cloudless skies; `sunpath` to sense clouds only in the sun path
        full_output: boolean
            True to get extra outputs (see below)
        times : Pandas DatetimeIndex or equivalent
            Vector of times. Required if meas_ghi and csky_ghi are not
            Pandas Series
        max_iterations: integer
            Maximum number of loop iterations to find a scaling factor for
            the clear-sky input fluxes
        rtol_alpha: float
            Precision of the maximum relative difference between scaling factors
            in successive loop interations. Defaults to 4 decimal positions

        Returns
        -------
        If full_output is False (default) returns a numpy array of bools, which
        is True for clear-sky and False otherwise.
        If full_output is True returns a numpy array of bools (same as before)
        and a dictionary with additional internal parameters (see code below)
        """

        from scipy.linalg import hankel
        from scipy.optimize import minimize_scalar

        OPTIMISATION_THRESH = 30  # W m-2
        UPPER_ALPHA_LIMIT = 1.5
        LOWER_ALPHA_LIMIT = 0.7

        if "ghi" not in self._sdf.columns:
            logger.warning("`ghi` column not found in dataframe. Cannot compute "
                           "Bright and Sun 2-class sky type.")
            return self._sdf.replace_data(1).iloc[:, 0].astype(np.int8).rename("bright_and_sun")

        if "dif" not in self._sdf.columns:
            logger.warning("`dif` column not found in dataframe. Cannot compute "
                           "Bright and Sun 2-class sky type.")
            return self._sdf.replace_data(1).iloc[:, 0].astype(np.int8).rename("bright_and_sun")

        sdf = normalize(self._sdf[["ghi", "dif"]])
        times = sdf.index
        meas_ghi = sdf.ghi.to_numpy()
        csky_ghi = sdf.clearsky.ghi.to_numpy().copy()
        meas_dif = sdf.dif.to_numpy()
        csky_dif = sdf.clearsky.dif.to_numpy().copy()
        times_tst = sdf.solpos.true_solar_time.to_numpy()
        sza = sdf.solpos.zenith.to_numpy()
        cosz = sdf.solpos.cosz.to_numpy()

        window_length_minutes = int(window_length_minutes)

        assert (len(times) == len(times_tst) == len(meas_ghi) ==
                len(csky_ghi) == len(meas_dif) == len(csky_dif) == len(sza))

        n_steps = len(times)
        meas_dni = np.divide(
            meas_ghi - meas_dif, cosz,
            out=np.full(n_steps, np.nan), where=sza < 89.
        )
        csky_dni = np.divide(
            csky_ghi - csky_dif, cosz,
            out=np.full(n_steps, np.nan), where=sza > 89.
        )

        one_minute = np.timedelta64(1, '60s')
        deltas = np.diff(times.values) / one_minute
        if len(np.unique(deltas)) > 1:
            raise ValueError(
                'the time difference between consecutive time steps must be ',
                'constant throughout the input time series')
        period_minutes = deltas[0]
        steps_per_window = int(window_length_minutes / period_minutes)

        #######################################
        # 1. DAY-BY-DAY CLEARSKY OPTIMIZATION #
        #######################################

        # ...it starts with a first-guess based on R&H (True => clear sky)
        initial_csd_flag = self.reno_and_hansen(var_diff_thresh=0.1, max_iterations=1)
        initial_csd_flag = initial_csd_flag.astype(bool)

        # ...and now, the daily "alpha" optimization
        dt64D = 'datetime64[D]'
        day_number = np.array(times_tst, dtype=dt64D).astype('int64')
        is_nat = day_number == np.array('NaT', dtype=dt64D).astype('int64')
        day_number[is_nat] = np.array(times[is_nat], dtype=dt64D).astype('int64')
        n_days = len(np.unique(day_number))

        alphas_ghi = np.ones(n_days)
        alphas_dni = np.ones(n_days)
        alphas_dif = np.ones(n_days)

        def match(a1, a2):
            return round(a1, rtol_alpha) == round(a2, rtol_alpha)

        @suppress_warnings(category=RuntimeWarning)
        def optimize(alpha, meas_clear, csky_clear):
            n_iter = 0
            prev_alpha = float('nan')
            while((n_iter < max_iterations) and not match(alpha, prev_alpha)):
                res = minimize_scalar(lambda a: np.sqrt(np.nanmean(meas_clear - a*csky_clear)**2))
                prev_alpha = alpha
                alpha = res.x
                n_iter += 1
            return min(max(alpha, LOWER_ALPHA_LIMIT), UPPER_ALPHA_LIMIT)

        for k, dn in enumerate(np.unique(day_number)):
            cur_day = dn == day_number

            cur_meas_ghi = meas_ghi[cur_day]
            cur_csky_ghi = csky_ghi[cur_day]
            cur_meas_dif = meas_dif[cur_day]
            cur_csky_dif = csky_dif[cur_day]
            cur_meas_dni = meas_dni[cur_day]
            cur_csky_dni = csky_dni[cur_day]
            cur_csd_flag = initial_csd_flag[cur_day]

            # ...optimisation lower limit threshold
            domain = ((cur_meas_dif < OPTIMISATION_THRESH) &
                      (cur_meas_ghi < OPTIMISATION_THRESH))
            cur_csd_flag[domain] = False  # False => cloudy

            if sum(cur_csd_flag) > 60:
                logger.debug(f'optimising for day {k}')
                alphas_ghi[k] = optimize(alphas_ghi[k], cur_meas_ghi[cur_csd_flag], cur_csky_ghi[cur_csd_flag])
                alphas_dif[k] = optimize(alphas_dif[k], cur_meas_dif[cur_csd_flag], cur_csky_dif[cur_csd_flag])
                alphas_dni[k] = optimize(alphas_dni[k], cur_meas_dni[cur_csd_flag], cur_csky_dni[cur_csd_flag])

            # apply the clear-sky correction factors
            csky_ghi[cur_day] = csky_ghi[cur_day]*alphas_ghi[k]
            csky_dif[cur_day] = csky_dif[cur_day]*alphas_dif[k]
            csky_dni[cur_day] = csky_dni[cur_day]*alphas_dni[k]

        # print(alphas_ghi)
        # print(alphas_dif)
        # print(alphas_dni)

        ###########################################
        # 2. TRI-COMPONENT MULTICRITERIA ANALYSIS #
        ###########################################

        H = hankel(np.arange(steps_per_window),
                np.arange(steps_per_window - 1, n_steps))

        # GHI analysis... ####################

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            meas_mean = np.nanmean(meas_ghi[H], axis=0)
            meas_max = np.nanmax(meas_ghi[H], axis=0)
            meas_diff = np.diff(meas_ghi[H], n=1, axis=0)
            meas_slope = np.diff(meas_ghi[H], n=1, axis=0) / period_minutes
            # ...and repeat for csky_ghi
            csky_mean = np.nanmean(csky_ghi[H], axis=0)
            csky_max = np.nanmax(csky_ghi[H], axis=0)
            csky_diff = np.diff(csky_ghi[H], n=1, axis=0)
            csky_slope = np.diff(csky_ghi[H], n=1, axis=0) / period_minutes
            zen = np.nanmean(sza[H], axis=0)

        meas_slope_nstd = np.full(meas_mean.shape, np.nan)
        universe = np.abs(np.nan_to_num(meas_mean)) > 1e-5
        meas_slope_nstd[universe] = \
            np.std(meas_slope, axis=0, ddof=1)[universe] / meas_mean[universe]
        meas_line_length = np.nansum(
            np.sqrt(meas_diff**2 + period_minutes**2), axis=0)
        csky_line_length = np.nansum(
            np.sqrt(csky_diff**2 + period_minutes**2), axis=0)

        # I took the coefficients for the limits from the Matlab code in github:
        # github.com/JamieMBright/csd-library/blob/master/models/BrightSun2020CSDc.m.
        # They do not all match the ones in Table 3 of Bright et al. (2020)

        def piecewise(*args):
            return np.piecewise(
                args[0],
                [arg[1] for arg in args[1:]],
                [arg[0] for arg in args[1:]]
            )

        c1_lim = piecewise(
            zen,
            (0.25, zen < 20.),
            (
                lambda x: np.interp(x, [20., 30.], [0.25, 0.125]),
                (zen >= 20.) & (zen < 30.)
            ),
            (
                lambda x: np.interp(x, [30., 90.], [0.125, 0.5]),
                zen >= 30.
            )
        )

        c2_lim = piecewise(
            zen,
            (0.25, zen < 20.),
            (
                lambda x: np.interp(x, [20., 30.], [0.25, 0.125]),
                (zen >= 20.) & (zen < 30.)
            ),
            (
                lambda x: np.interp(x, [30., 90.], [0.125, 0.5]),
                zen >= 30.
            )
        )

        c3_lim = piecewise(
            zen,
            (
                lambda x: np.interp(x, [0., 30.], [-7., -0.5]),
                (zen >= 0.) & (zen < 30.)
            ),
            (
                -0.5,
                zen >= 30.
            )
        )

        c5_lim = piecewise(
            zen,
            (
                lambda x: np.interp(x, [0., 30.], [45., 15.]),
                (zen >= 0.) & (zen < 30.)
            ),
            (
                15.,
                zen >= 30.
            )
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rel_diff = np.abs(meas_mean - csky_mean)/csky_mean
            cond1 = np.nan_to_num(rel_diff, nan=np.inf) < c1_lim  # if True => clear
            rel_diff = np.abs(meas_max - csky_max)/csky_max
            cond2 = np.nan_to_num(rel_diff, nan=np.inf) < c2_lim  # if True => clear
            rel_diff = (meas_line_length - csky_line_length)/csky_line_length
            cond3 = (
                (np.nan_to_num(rel_diff, nan=-np.inf) > c3_lim) &
                (np.nan_to_num(rel_diff, nan=np.inf) < np.abs(c3_lim))  # True => clear
            )
            cond4 = np.nan_to_num(meas_slope_nstd, nan=np.inf) < 0.4  # True => clear
            rel_diff = np.nanmax(np.abs(meas_slope - csky_slope), axis=0)
            cond5 = np.nan_to_num(rel_diff, nan=np.inf) < c5_lim  # if True => clear
            cond6 = (csky_mean != 0.) & ~np.isnan(csky_mean)

        # ...clear-sky flag for GHI: True => clear sky; False => cloudy sky
        conds = cond1 & cond2 & cond3 & cond4 & cond5 & cond6
        clearsky_ghi_flag = np.full_like(meas_ghi, False, dtype='bool')
        clearsky_ghi_flag[np.unique(H[:, conds])] = True

        # DIF analysis... ####################

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            meas_mean = np.nanmean(meas_dif[H], axis=0)
            meas_max = np.nanmax(meas_dif[H], axis=0)
            meas_diff = np.diff(meas_dif[H], n=1, axis=0)
            meas_slope = np.diff(meas_dif[H], n=1, axis=0) / period_minutes
            # ...and repeat for csky_dif
            csky_mean = np.nanmean(csky_dif[H], axis=0)
            csky_max = np.nanmax(csky_dif[H], axis=0)
            csky_diff = np.diff(csky_dif[H], n=1, axis=0)
            csky_slope = np.diff(csky_dif[H], n=1, axis=0) / period_minutes

        meas_slope_nstd = np.full(meas_mean.shape, np.nan)
        universe = np.abs(np.nan_to_num(meas_mean)) > 1e-5
        meas_slope_nstd[universe] = \
            np.std(meas_slope, axis=0, ddof=1)[universe] / meas_mean[universe]
        meas_line_length = np.nansum(
            np.sqrt(meas_diff**2 + period_minutes**2), axis=0)
        csky_line_length = np.nansum(
            np.sqrt(csky_diff**2 + period_minutes**2), axis=0)

        # I took the coefficients for the limits from the Matlab code in github:
        # github.com/JamieMBright/csd-library/blob/master/models/BrightSun2020CSDc.m.
        # They do not all match the ones in Table 3 of Bright et al. (2020)

        c1_lim = piecewise(
            zen,
            (0.25, zen < 20.),
            (
                lambda x: np.interp(x, [20., 30.], [0.25, 0.5]),
                (zen >= 20.) & (zen < 30.)
            ),
            (
                0.5,
                zen >= 30.
            )
        )

        c2_lim = piecewise(
            zen,
            (
                0.25,
                zen < 20.
            ),
            (
                lambda x: np.interp(x, [20., 30.], [0.25, 0.5]),
                (zen >= 20.) & (zen < 30.)
            ),
            (
                0.5,
                zen >= 30.
            )
        )

        c3_lim = piecewise(
            zen,
            (
                lambda x: np.interp(x, [0., 30.], [-6., -1.7]),
                (zen >= 0.) & (zen < 30.)
            ),
            (
                -1.7,
                zen >= 30.
            )
        )

        c5_lim = piecewise(
            zen,
            (
                lambda x: np.interp(x, [0., 30.], [24., 8.]),
                (zen >= 0.) & (zen < 30.)
            ),
            (
                8.,
                zen >= 30.
            )
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rel_diff = np.abs(meas_mean - csky_mean)/csky_mean
            cond1 = np.nan_to_num(rel_diff, nan=np.inf) < c1_lim  # if True => clear
            rel_diff = np.abs(meas_max - csky_max)/csky_max
            cond2 = np.nan_to_num(rel_diff, nan=np.inf) < c2_lim  # if True => clear
            rel_diff = (meas_line_length - csky_line_length)/csky_line_length
            cond3 = (
                (np.nan_to_num(rel_diff, nan=-np.inf) > c3_lim) &
                (np.nan_to_num(rel_diff, nan=np.inf) < np.abs(c3_lim))  # True => clear
            )
            cond4 = np.nan_to_num(meas_slope_nstd, nan=np.inf) < 0.2  # True => clear
            rel_diff = np.nanmax(np.abs(meas_slope - csky_slope), axis=0)
            cond5 = np.nan_to_num(rel_diff, nan=np.inf) < c5_lim  # if True => clear
            cond6 = (csky_mean != 0.) & ~np.isnan(csky_mean)

        # ...clear-sky flag for DIF: True => clear sky; False => cloudy sky
        conds = cond1 & cond2 & cond3 & cond4 & cond5 & cond6
        clearsky_dif_flag = np.full_like(meas_dif, False, dtype='bool')
        clearsky_dif_flag[np.unique(H[:, conds])] = True

        # DNI analysis... ####################

        kcb = np.divide(
            meas_dni, csky_dni,
            out=np.full(meas_dni.shape, np.nan), where=csky_dni > 0.
        )
        zen = np.degrees(np.arccos(cosz))
        kcb_lim = np.full(zen.shape, 0.9)
        kcb_lim[zen >= 30.] = -0.0067*np.minimum(zen[zen >= 30.], 90.)+1.1

        # ...clear-sky flag for DNI: True => clear sky; False => cloudy sky
        clearsky_dni_flag = np.full_like(meas_dni, False, dtype='bool')
        clearsky_dni_flag[kcb > kcb_lim] = True

        # ...clear-sky flag after the tri-component analysis:
        # True => clear; False => cloudy
        clearsky_3comp_flag = (
            clearsky_ghi_flag & clearsky_dif_flag & clearsky_dni_flag)

        if csd == 'sunpath':

            clearsky_flag = clearsky_3comp_flag

            result = SolarSeries(
                data=clearsky_flag,
                index=times,
                latitude=self._sdf.latitude,
                longitude=self._sdf.longitude,
                elevation=self._sdf.elevation,
                custom_metadata=self._sdf.custom_metadata,
                dtype=np.int8)

            # keep only the original timestamps
            result = result.reindex(self._sdf.index)

            # restore the original index, in particular, if it was tz-naive
            result.index = self._sdf.index
            result.name = f"bright_and_sun/{csd}"

            return result

        #######################
        # 3. DURATION FILTERS #
        #######################

        # 1st duration filter:
        # duration filter looking ahead and behind for 45 minutes. Should there
        # not have been a continuous CSD for an hour, then all instances are
        # rejected
        window_size, tolerance = 90, 10
        H = hankel(np.arange(window_size), np.arange(window_size - 1, n_steps))
        dur1_cond = np.nansum(np.where(clearsky_3comp_flag, 0, 1)[H], axis=0)
        clearsky_d1_flag = np.full_like(meas_ghi, True, dtype='bool')
        start = window_size//2
        end = start + len(dur1_cond)
        clearsky_d1_flag[start: end][dur1_cond] = False

        # near sunrise and sunset (assumed as zen=85deg), the filter is relaxed. I
        # take a shortcut here, with respect to the original Matlab code
        nighttime_flag = np.full_like(meas_ghi, False, dtype='bool')
        nighttime_cond = np.sum(np.degrees(np.arccos(cosz))[H] > 85., axis=0) > 0
        start = window_size//2
        end = start + len(nighttime_cond)
        nighttime_flag[start: end] = nighttime_cond

        clearsky_d1_flag[nighttime_flag] = True

        # 2nd duration filter:
        window_size, tolerance = 30, 0
        H = hankel(np.arange(window_size), np.arange(window_size - 1, n_steps))
        dur2_cond = np.nansum(
            np.where(clearsky_3comp_flag, 0, 1)[H], axis=0) > tolerance
        clearsky_d2_flag = np.full_like(meas_ghi, True, dtype='bool')
        start = window_size//2
        end = start + len(dur2_cond)
        clearsky_d2_flag[start: end][dur2_cond] = False

        # 3rd duration filter:
        window_size, tolerance = 10, 2
        H = hankel(np.arange(window_size), np.arange(window_size - 1, n_steps))
        dur3_cond = np.nansum(
            np.where(clearsky_3comp_flag, 0, 1)[H], axis=0) > tolerance
        clearsky_d3_flag = np.full_like(meas_ghi, True, dtype='bool')
        clearsky_d3_flag[window_size//2:window_size//2 + len(dur3_cond)] = False

        clearsky_d2_flag[clearsky_d3_flag & nighttime_flag] = True

        # combine the 3-component flag and the "duration" flags
        clearsky_flag = clearsky_3comp_flag & clearsky_d1_flag & clearsky_d2_flag

        result = SolarSeries(
            data=clearsky_flag,
            index=times,
            latitude=self._sdf.latitude,
            longitude=self._sdf.longitude,
            elevation=self._sdf.elevation,
            custom_metadata=self._sdf.custom_metadata,
            dtype=np.int8)

        # keep only the original timestamps
        result = result.reindex(self._sdf.index)

        # restore the original index, in particular, if it was tz-naive
        result.index = self._sdf.index
        result.name = f"bright_and_sun/{csd}"

        return result
