"""Accessors to evaluate pv yield."""

from typing import Literal

import numpy as np
import pandas as pd
import pvlib
from loguru import logger

from ..base import SolarSeries, SolarDataFrame
from ..helpers import infer_time_step, normalize

logger.disable(__name__)
logger = logger.opt(colors=True)


class PVAccessor:
    """Accessor for PV yield evaluation.

    Examples
    --------
    >>> sdf.pv.yield_dc(...)
    >>> sdf.pv.yield_ac(...)
    >>> sdf.pv.clipping_losses(...)
    >>> sdf.pv.optimal_dc_to_ac_ratio(...)
    """

    def __init__(self, sdf_obj):
        self._sdf: SolarDataFrame = self._validate(sdf_obj)

    @staticmethod
    def _validate(obj):
        if not isinstance(obj, SolarDataFrame):
            name = obj.__class__.__name__
            raise AttributeError(f"required a SolarDataFrame instance. Got {name}")
        obj = obj.resolve_closure()  # compute solar components using closure where possible
        if not {"ghi", "dni", "dif"}.issubset(obj.columns):
            raise AttributeError(f"required a SolarDataFrame with columns {{'ghi', 'dni', 'dif'}}, but got {obj.columns}")
        return normalize(obj)  # ensure complete days for PV yield calculations

    def poa_irradiance(
        self,
        tracking: Literal["fixed", "fixed_optimal", "singleaxis", "dualaxis"] = "singleaxis",
        aoi_losses: bool = True,
        tracking_kwargs: dict = None,
        transposition_kwargs: dict = None,
    ) -> SolarDataFrame:
        """Transposition of solar irradiance to the PV plane of array.

        It requires `ghi`, `dni` and `dif` columns. The calculations are delegated to the `pvlib` library.

        Parameters
        ----------
        tracking: str
            type of tracking system to use for calculating the plane-of-array (POA) irradiance. Options are:
            - "fixed": fixed tilt system, with the tilt and azimuth angles specified in `tracking_kwargs` as
                `poa_tilt` and `poa_azimuth`, respectively. The tilt angle is the angle between the plane of the
                PV array and the horizontal plane, while the azimuth angle is the compass direction that the PV
                array faces (0° for north, 90° for east, 180° for south, and 270° for west).
            - "fixed_optimal": fixed tilt system with the optimal tilt angle calculated based on the latitude
                of the location. The optimal tilt angle is calculated using a simple empirical formula that provides
                a good approximation for many locations: `poa_tilt = 0.87*latitude` for latitudes between -25° and
                25°, `poa_tilt = 0.76*latitude + 3.1` for latitudes between -50° and 50°, and a fixed tilt of 40°
                for latitudes outside this range. The azimuth angle is set to 180° (south-facing) by default.
            - "singleaxis": single-axis tracking system, with the tracking parameters specified in `tracking_kwargs`
                as `axis_tilt`, `axis_azimuth`, `max_angle`, `backtrack`, and `gcr`. The single-axis tracking system
                rotates around a single axis to follow the sun's movement, which can increase the energy yield
                compared to a fixed tilt system. The `axis_tilt` is the angle of the rotation axis relative to the
                horizontal plane, while the `axis_azimuth` is the compass direction of the rotation axis. The
                `max_angle` is the maximum rotation angle of the tracker, while `backtrack` indicates whether to use
                backtracking to avoid shading between rows of panels. The `gcr` (ground coverage ratio) is the ratio
                of the area covered by the PV panels to the total ground area.
            - "dualaxis": dual-axis tracking system, which can rotate around two axes to follow the sun's movement
                more accurately. In this case, the plane-of-array (POA) tilt and azimuth angles are calculated based
                on the solar zenith and azimuth angles, resulting in a POA that is always perpendicular to the sun's
                rays. This type of tracking system can provide the highest energy yield, but it is also more complex
                and expensive than fixed or single-axis tracking systems. The dual-axis tracking system is particularly
                beneficial in locations with high solar variability or for applications that require maximizing energy
                yield, such as in concentrated solar power (CSP) systems or for certain types of PV installations
                where space is limited and maximizing energy production is critical.
        aoi_losses: bool
            whether to apply the Martin-Ruiz incidence angle modifier (IAM) correction to account for the reduction
            in effective irradiance on the PV modules at high angles of incidence. The IAM correction reduces the
            effective irradiance on the PV modules when the angle of incidence of the sunlight is large, which can
            occur during early morning, late afternoon, or in locations with high solar zenith angles. Applying the
            IAM correction can provide a more accurate estimation of the DC power output, especially for fixed tilt
            systems or for locations with high solar variability. The Martin-Ruiz model is a widely used empirical
            model for calculating the IAM and is based on measurements of the performance of PV modules at different
            angles of incidence.
        tracking_kwargs: dict
            keyword arguments for the tracking system, such as `poa_tilt` and `poa_azimuth`, which must be provided
            for fixed tilt systems (`tracking='fixed'`), or `axis_tilt`, `axis_azimuth`, `max_angle`, `backtrack`,
            and `gcr` for single-axis tracking systems (`tracking='single-axis'`).
            The default values for the single-axis parameters are `axis_tilt=0.`, `axis_azimuth=180.`, `max_angle=60.`,
            `backtrack=True`, and `gcr=0.4`, which correspond to a common single-axis tracking configuration with a
            horizontal rotation axis (sometimes referred to as HSAT --horizontal single-axis tracker--) and a ground
            coverage ratio of 0.4, which is typical for commercial PV installations. These default values can be
            overridden by providing the desired values in `tracking_kwargs`.
            For dual-axis tracking systems (`tracking='dual-axis'`), no additional parameters are required as the POA
            tilt and azimuth are calculated based on the solar position.
        transposition_kwargs: dict
            keyword arguments to select the transposition model to be used to calculate the POA irradiance. It can
            affect the accuracy of the POA irradiance calculation, especially under certain sky conditions (e.g., cloudy
            vs. clear skies). The model is selected with the key `model`, whose possible values are `isotropic` (the
            default), `klucher`, `haydavies`, `reindl`, `king`, `perez` and `perez-driesse`. Some of the model choices
            accept additional parameters to be passed also in transposition_kwargs, such as, "perez_model" if the selected
            model is `perez`, or "dni_extra" if the selected model is one of `haydavies`, `reindl`, `perez` or
            `perez-driesse`.
        
        Returns
        -------
        SolarDataFrame
            a new SolarDataFrame with the same index as the original but with the following columns:
            - `tilt`: plane-of-array tilt angle (degrees)
            - `azimuth`: plane-of-array azimuth angle (degrees)
            - `aoi`: angle of incidence of the sunlight on the plane of array (degrees)
            - `direct`: direct component of the plane-of-array irradiance (W m-2)
            - `sky_diffuse`: sky diffuse component of the plane-of-array irradiance (W m-2)
            - `ground_diffuse`: ground diffuse component of the plane-of-array irradiance (W m-2)
            - `diffuse`: total diffuse component of the plane-of-array irradiance (W m-2)
            - `global`: total plane-of-array irradiance (W m-2)

        Notes
        -----
        For more details see the `pvlib` documentation for the `get_total_irradiance` function, which is used to perform
        the transposition calculations, and the `iam.martin_ruiz` and `iam.martin_ruiz_diffuse` functions, which are used
        to apply the incidence angle modifier (IAM) correction if `aoi_losses` is set to True.

        https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.irradiance.get_total_irradiance.html#pvlib-irradiance-get-total-irradiance

        """

        if tracking not in ("fixed", "fixed_optimal", "singleaxis", "dualaxis"):
            raise AssertionError(f"unknown option {tracking=}")

        sdf = self._sdf.resolve_closure()

        tracking_kwargs = tracking_kwargs or {}
        tracking_kwargs.setdefault("axis_tilt", 0.)  # horizontal axis
        tracking_kwargs.setdefault("axis_azimuth", 180.)  # south-facing
        tracking_kwargs.setdefault("max_angle", 60.)  # a common maximum rotation
        tracking_kwargs.setdefault("backtrack", True)  # backtrack for a typical c-Si array
        tracking_kwargs.setdefault("gcr", 0.4)  # common ground coverage ratio

        transposition_kwargs = transposition_kwargs or {}
        transposition_kwargs.setdefault("model", "isotropic")  # the simplest transposition approach

        poa = pd.DataFrame(index=sdf.index, dtype=float)  # container for output results

        # t r a c k i n g: compute poa tilt, azimuth and angle of incidence (aoi) from the tracker configuration

        if tracking == "fixed":
            if "poa_tilt" not in tracking_kwargs:
                raise ValueError("`poa_tilt` required in `tracking_kwargs` for tracking='fixed'")
            if "poa_azimuth" not in tracking_kwargs:
                raise ValueError("`poa_azimuth` required in `tracking_kwargs` for tracking='fixed'")
            poa_tilt = tracking_kwargs["poa_tilt"]
            poa_azimuth = tracking_kwargs["poa_azimuth"]
            aoi = pvlib.irradiance.aoi(poa_tilt, poa_azimuth, sdf.solpos.zenith, sdf.solpos.azimuth)
            poa = poa.assign(tilt=poa_tilt, azimuth=poa_azimuth, aoi=aoi)

        if tracking == "fixed_optimal":
            poa_tilt = (0.87*sdf.latitude if -25 < sdf.latitude < 25
                        else (0.76*sdf.latitude+3.1 if -50 < sdf.latitude < 50 else 40.))
            poa_azimuth = 0.  # southern
            aoi = pvlib.irradiance.aoi(poa_tilt, poa_azimuth, sdf.solpos.zenith, sdf.solpos.azimuth)
            poa = poa.assign(tilt=poa_tilt, azimuth=poa_azimuth, aoi=aoi)

        if tracking == "singleaxis":
            tracking_geometry = pvlib.tracking.singleaxis(sdf.solpos.zenith, sdf.solpos.azimuth, **tracking_kwargs)
            poa_tilt = tracking_geometry["surface_tilt"]
            poa_azimuth = tracking_geometry["surface_azimuth"]
            aoi = tracking_geometry["aoi"]
            poa = poa.assign(tilt=poa_tilt, azimuth=poa_azimuth, aoi=aoi)

        if tracking == "dualaxis":
            poa_tilt = sdf.solpos.zenith.clip(lower=0, upper=90)
            poa_azimuth = sdf.solpos.azimuth
            poa = poa.assign(tilt=poa_tilt, azimuth=poa_azimuth, aoi=0.)

        # t r a n s p o s i t i o n

        transposed_irradiance = pvlib.irradiance.get_total_irradiance(
            surface_tilt=poa.tilt,
            surface_azimuth=poa.azimuth,
            solar_zenith=sdf.solpos.zenith,
            solar_azimuth=sdf.solpos.azimuth,
            ghi=sdf.ghi,
            dni=sdf.dni,
            dhi=sdf.dif,
            **transposition_kwargs)

        poa = poa.assign(**transposed_irradiance.rename(columns=lambda name: name.removeprefix("poa_")))

        # i n c i d e n c e   a n g l e   m o d i f i e r   ( I A M )   c o r r e c t i o n

        if aoi_losses:
            iam_direct = pvlib.iam.martin_ruiz(poa.aoi)
            poa["direct"] = poa["direct"] * iam_direct
            iam_sky, iam_ground = pvlib.iam.martin_ruiz_diffuse(poa.tilt)
            poa["sky_diffuse"] = poa["sky_diffuse"] * iam_sky
            poa["ground_diffuse"] = poa["ground_diffuse"] * iam_ground

        # N.B. I have noticed that sky/ground diffuse provides negative values sometimes. This can be due to
        # inconsistencies in the transposition models and should have been screened in pvlib, but it is not.
        # I am simply clipping it here.
        poa["sky_diffuse"] = poa["sky_diffuse"].clip(lower=0)
        poa["ground_diffuse"] = poa["ground_diffuse"].clip(lower=0)
        poa["diffuse"] = poa["sky_diffuse"] + poa["ground_diffuse"]
        poa["direct"] = poa["direct"].clip(lower=0)
        poa["global"] = poa["diffuse"] + poa["direct"]

        sdf = sdf.replace_data(poa)
        sdf.custom_metadata.update({"tracking": tracking_kwargs, "transposition": transposition_kwargs})
        return sdf

    def yield_dc(
        self,
        # options for PV modelling
        p_dc_peak: float = 1.,  # power capacity in Wp
        dc_model: Literal["huld", "pvwatts"] = "huld",
        dc_model_kwargs: dict | None = None,
        # additional atmospheric data
        temperature: pd.Series | str | float | None = None,
        wind_speed: pd.Series | str | float | None = None,  # m s-1
        # plane-of-array (POA) irradiance calculation options
        poa_irradiance_kwargs: dict | None = None,
        units: Literal["W", "Wh"] = "W",
        full_output: bool = False
    ) -> SolarSeries | SolarDataFrame:

        """Calculate the DC power yield of a PV system.

        Parameters
        ----------
        p_dc_peak : float
            Installed peak capacity in Watts-peak. Default is 1 Wp.
        dc_model : str
            Model to convert plane-of-array (POA) irradiance and cell temperature into DC power. Options are:
            - "pvwatts": the PVWatts model, which is a simple empirical model based on the performance of a
                large number of PV systems. It is widely used for its simplicity and reasonable accuracy for many
                applications. It estimates the DC power output based on the plane-of-array irradiance, the cell
                temperature, and the installed DC capacity, using a simple linear model with a temperature
                coefficient.
            - "huld": the Huld model, which is a more detailed empirical model that accounts for the non-linear
                effects of irradiance and temperature on the DC power output. It is based on the performance of a
                large number of PV systems and provides a more accurate estimation of the DC power output, especially
                under low irradiance and high temperature conditions. It requires more parameters than the PVWatts
                model, such as the cell type and the version of the model to use (e.g., "pvgis6" for the latest
                version). The Huld model is particularly useful for applications that require a more accurate
                estimation of the DC power output, such as performance ratio calculations or detailed energy yield
                assessments.
            The Faiman's cell temperature model is used. See `pvlib`.
        dc_model_kwargs : dict
            Additional kwargs to be passed to the "dc_model". See `pvlib.pvsystem.pvwatts_dc` and `pvlib.pvarray.huld`
            for details. For `dc_model="huld"`, default values `cell_type="cSi"` and `k_version="pvgis6"` are used.
        temperature : pd.Series, str, float or None. Default is None.
            Air temperature in deg C. If it is str, it should be a column name in the SolarDataFrame. If it is None,
            a default temperature value of 25 deg C is used.
        wind_speed : pd.Series, str, float or None. Default is None.
            Wind speed in m s-1. If it is str, it should be a column name in the SolarDataFrame. If it is None, a
            default wind speed value of 1 m s-1 is used.
        poa_irradiance_kwargs : dict | None. Default is None.
            Additional kwargs to be passed to the `poa_irradiance` method. See `poa_irradiance` for details.
        units : str
            Units for the output power. Options are "W" for Watts (instantaneous power) and "Wh" for Watt-hours (energy).
            If "Wh" is selected, the output power is multiplied by the time step inferred from the index of the
            SolarDataFrame, so it represents the energy produced during each time step. Default is "W".
        full_output : bool
            Include the outputs from `poa_irradiance`. Default is False.

        Returns
        -------
        SolarSeries or SolarDataFrame
            a new SolarDataFrame with the same index as the original but with the following columns:
            - `pdc`: DC power output of the PV system in Watts
            - the output columns of `poa_irradiance` if `full_output=True`.
        """

        DEFAULT_TEMPERATURE = 25.  # deg C
        DEFAULT_WIND_SPEED = 1.  # m s-1

        if temperature is None:
            temperature = DEFAULT_TEMPERATURE

        if isinstance(temperature, str):
            if temperature in self._sdf.columns:
                temperature = self._sdf[temperature]
            else:
                temperature = DEFAULT_TEMPERATURE
                logger.warning (f"temperature column '{temperature}' not found in data. "
                                f"Set to the default value {DEFAULT_TEMPERATURE} deg C.")

        if wind_speed is None:
            wind_speed = DEFAULT_WIND_SPEED

        if isinstance(wind_speed, str):
            if wind_speed in self._sdf.columns:
                wind_speed = self._sdf[wind_speed]
            else:
                wind_speed = DEFAULT_WIND_SPEED
                logger.warning(f"wind_speed column '{wind_speed}' not found in data. "
                               f"Set to the default value {DEFAULT_WIND_SPEED} m s-1.")

        poa_irradiance_kwargs = poa_irradiance_kwargs or {}
        poa = self.poa_irradiance(**poa_irradiance_kwargs)
        poa = poa.assign(temperature=temperature, wind_speed=wind_speed)

        # to detect potential RuntimeWarning: invalid value encountered in log
        if (illegal := poa["global"].le(-1e-6) & self._sdf.solpos.zenith.lt(90.)).any():
            logger.warning(f"Illegal data found.\ndata:\n{self._sdf.loc[illegal]}\npoa:\n{poa.loc[illegal]}")

        temp_cell = pvlib.temperature.faiman(
            poa_global=poa["global"],  # total incident irradiance, W m-2
            temp_air=poa["temperature"],  # ambient dry bulb temperature, degC
            wind_speed=poa["wind_speed"])  # wind speed, m s-1

        if dc_model == "pvwatts":
            pdc = pvlib.pvsystem.pvwatts_dc(
                effective_irradiance=poa["global"],
                temp_cell=temp_cell,
                pdc0=p_dc_peak,  # power of the modules at STC (i.e., 1000 W m-2 and 25 degC), W
                **(dc_model_kwargs or {}),
            ).clip(lower=0.)

        if dc_model == "huld":
            dc_model_kwargs = dc_model_kwargs or {}
            dc_model_kwargs.setdefault("cell_type", "cSi")
            dc_model_kwargs.setdefault("k_version", "pvgis6")
            pdc = pvlib.pvarray.huld(
                effective_irradiance=poa["global"],  # irradiance to be converted, W m-2
                temp_mod=temp_cell,  # module back-surface temperature, degC
                pdc0=p_dc_peak,  # power of the modules at STC (i.e., 1000 W m-2 and 25 degC), W
                **dc_model_kwargs
            ).clip(lower=0)

        if units == "Wh":
            time_step_seconds = infer_time_step(self._sdf).total_seconds()
            time_step_hours = time_step_seconds / 3600.
            pdc = pdc.mul(time_step_hours).rename("pdc")

        pdc = poa.replace_data(pdc).iloc[:, 0].rename("pdc")
        pvsystem = {
            "dc_model": dc_model,
            "dc_model_kwargs": dc_model_kwargs,
            "p_dc_peak": p_dc_peak,
            "pdc_units": units,
            "poa_irradiance_kwargs": poa_irradiance_kwargs}
        pdc.custom_metadata.update({"pvsystem": pvsystem})
        if full_output:
            return pdc.to_frame().join(poa)
        return pdc

    def yield_ac(
        self,
        dc_to_ac_ratio: float = 1.0,
        inverter_effic: float = 0.96,
        yield_dc_kwargs: dict | None = None,
        units: Literal["W", "Wh"] = "W",
        full_output: bool = False
    ) -> SolarSeries | SolarDataFrame:

        """Calculate the AC power yield of a PV system.

        For simplicity, the `pvwatts` inverter is assumed (see Notes below). For more precise simulation of
        specific inverters, consider using the Sandia or ADR inverters included also in `pvlib`. Unfortunately,
        they are not supported here.

        Parameters
        ----------
        dc_to_ac_ratio: float
            DC/AC ratio of the PV system, i.e., the ratio between the DC power capacity (the installed power
            of the PV modules) and the AC power capacity (the nominal power of the inverter). Hence, the size
            of the DC-to-AC inverter is defined with respect to the DC capacity using this ratio. It is an
            important design parameter in PV systems, as it determines how much of the DC power can be
            converted into AC power. A higher DC/AC ratio means that the inverter is "undersized" relative
            to the DC capacity, which can lead to clipping losses during periods of high irradiance. However,
            it can also reduce costs, as inverters are typically more expensive than PV modules. The optimal
            DC/AC ratio depends on various factors, including the cost of the inverter and modules, the expected
            irradiance levels, and the specific design goals of the PV system.
        inverter_effic: float
            nominal efficiency of the inverter (between 0 and 1)
        yield_dc_kwargs: dict | None
            Additional kwargs to be passed to the `yield_dc` method. See `yield_dc` for details.
        units: str
            Units for the output power. Options are "W" for Watts (instantaneous power) and "Wh" for Watt-hours (energy).
        full_output: bool
            Include the outputs from `yield_dc`. Default is False.

        Returns
        -------
        SolarSeries or SolarDataFrame
            a new SolarDataFrame with the same index as the original but with the following columns:
            - `pac`: AC power output of the PV system in Watts
            - the output columns of `yield_dc` if `full_output=True`.

        Notes
        -----
        The AC power output is calculated by applying the `pvlib`'s `pvwatts` inverter model to the DC power output
        obtained from the `yield_dc` method. Besides the input DC power, the `pvwatts` inverter requires the DC power
        at which the inverter reaches its AC power limit and the nominal efficiency of the inverter. The former is
        calculated based on the installed DC capacity and the specified DC/AC ratio. It represents an upper threshold
        for the DC power input to the inverter. If the DC power input exceeds this threshold, the inverter will "clip"
        the output power to its maximum AC power capacity.
        
        Typically, however, what is known at design time is the nominal DC power capacity of the modules and the DC/AC
        ratio is used to determine the nominal AC power capacity of the inverter (i.e., the maximum AC power output it
        can deliver). For example, for a DC power capacity of 1000 Wp and a DC/AC ratio of 1.2, the nominal AC power
        capacity of the inverter would be 833.3 W. These 833.3 W represent the maximum AC power output the inverter can
        deliver, that is, greater values are "clipped". To achieve this behavior with the `pvwatts` inverter, the nomial
        AC power of the inverter still has to be divided by the inverter efficiency. This is the approach used here.

        Why is `dc_to_ac_ratio` typically greater than one (i.e., the inverter is undersized relative to the DC capacity)?
        The main reason is cost. Inverters are typically more expensive than PV modules, and oversizing the inverter to
        match the DC capacity would increase the overall system cost. By undersizing the inverter, the system can be more
        cost-effective, even though it may result in some clipping losses (see `clipping_losses`) during periods of high
        irradiance. The optimal DC/AC ratio (see `optimal_dc_ac_ratio`) depends on various factors, including the cost of
        the inverter and modules, the expected irradiance levels, and the specific design goals of the PV system. Values
        of dc_to_ac_ratio between 1.2 and 1.4 are common in commercial PV systems, as they provide a good balance between
        cost and performance.
        """

        yield_dc_kwargs = yield_dc_kwargs or {}
        yield_dc_kwargs.update({"units": "W"})
        pdc = self.yield_dc(**yield_dc_kwargs)  # in Watts
        p_dc_peak = float(pdc.custom_metadata["pvsystem"]["p_dc_peak"])

        if isinstance(pdc, pd.Series):
            pdc = pdc.to_frame()

        p_ac_peak = p_dc_peak / dc_to_ac_ratio  # inverter power limit, in Watts-peak
        pdc0 = p_ac_peak / inverter_effic  # input DC power at which the inverter outputs its maximum AC power, in Watts
        pac = pvlib.inverter.pvwatts(  # in Watts
            pdc=pdc["pdc"],  # DC power input to the inverter, W
            pdc0=pdc0,
            eta_inv_nom=inverter_effic)

        if units == "Wh":
            time_step_seconds = infer_time_step(self._sdf).total_seconds()
            time_step_hours = time_step_seconds / 3600.
            pdc["pdc"] = pdc["pdc"].mul(time_step_hours)
            pac = pac.mul(time_step_hours)

        pac = pdc.replace_data(pac).iloc[:, 0].rename("pac")
        pvsystem = pac.custom_metadata["pvsystem"]
        pvsystem = pvsystem | {
            "inverter": "pvwatts",
            "dc_to_ac_ratio": dc_to_ac_ratio,
            "inverter_effic": inverter_effic,
            "pac_units": units}
        pac.custom_metadata.update({"pvsystem": pvsystem})
        if full_output:
            return pac.to_frame().join(pdc)
        return pac

    def clipping_losses(
        self,
        dc_to_ac_ratio: np.ndarray[tuple[int]] | float | None = None,
        time_series: bool = False,
        units: Literal["W", "fraction"] = "fraction",
        yield_dc_kwargs: dict | None = None,
        inverter_effic: float = 0.96,
        method: Literal["integral", "explicit"] = "integral",
        integral_bins: np.ndarray[tuple[int]] | int = 200,
    ) -> float | pd.Series | SolarDataFrame:
        r"""Calculate the clipping losses of a PV system assuming the pvlib's PVWatts inverter model.

        Parameters
        ----------
        dc_to_ac_ratio: float or array-like
            DC/AC ratio of the PV system. If None, a default range of values from 1.0 to 1.8
            (inclusive) is used.
        time_series: bool, default False
            If True, returns the time series of clipping losses instead of the total value. If True,
            `dc_to_ac_ratio` must be a single value.
        units: str, default "fraction"
            Units for the output clipping losses. Options are "fraction" for the fraction of DC
            power yield lost due to clipping, and "W" for the average DC power lost due to clipping.
        yield_dc_kwargs: dict, default None
            Keyword arguments to be passed to the `yield_dc` method.
        inverter_effic: float, default 0.96
            Inverter efficiency.
        method: str, default "integral"
            Method to use for calculating clipping losses. Options are "integral" and "explicit".
        integral_bins: int or array-like, default 200
            Number of bins to evaluate the integral for the integral method.
        
        Returns
        -------
        float or pd.Series or SolarDataFrame
            If `time_series` is True, a SolarDataFrame is returned with the time series of clipping
            losses, DC power and AC power. Otherwise, it returns clipping losses as a fraction of the
            total DC energy yield. If `dc_to_ac_ratio` is a single value, a float is returned. If
            `dc_to_ac_ratio` is an array-like, a pd.Series is returned with the DC/AC ratios as the index.

        Notes
        -----
        The `explicit` method is based on the Michelli et al. (doi: 
        [10.1016/j.renene.2024.120317](https://doi.org/10.1016/j.renene.2024.120317)) approach:

        ```math
        C_L = E_{DC} \eta_{inv} - E^{peak}_{AC}
        ```

        where $E_{DC}$ is the DC energy yield, $\eta_{inv}$ is the inverter's nominal efficiency,
        $E^{peak}_{AC} = \frac{E_{DC}}{\tau}$ is the AC energy yield at the inverter's peak power limit,
        and $\tau$ is the DC/AC ratio.

        The `integral` method evaluates clipping losses making explicit their relation with the probability
        density function (PDF) of the DC power output, and so also to that of the plane-of-array (POA)
        irradiance. Thus, it provides direct evidence of the impact of solar irradiance characteristics
        on clipping losses.
    
        ```math
        C_L = \int_{P_{AC}^{peak}}^{\infty} (P_{DC} \eta_{inv} - P_{AC}^{peak}) f(P_{DC}) dP_{DC}
        ```
        """

        if dc_to_ac_ratio is None:
            dc_to_ac_ratio = np.linspace(1., 2., 51)

        if isinstance(dc_to_ac_ratio, (int, float)):
            dc_to_ac_ratio = np.array([dc_to_ac_ratio])

        yield_dc_kwargs = yield_dc_kwargs or {}
        yield_dc_kwargs.update({"units": "W", "full_output": False})
        pdc = self.yield_dc(**yield_dc_kwargs)
        p_dc_peak = float(pdc.custom_metadata["pvsystem"]["p_dc_peak"])

        max_sza = 180.  # deg
        sza = pdc.solpos.zenith
        diurnal = sza.lt(max_sza)

        def compute_clipping_losses_numpy(
            pdc: np.ndarray[tuple[int]]
        ) -> tuple[np.ndarray, np.ndarray]:
            """Compute the clipping losses for a given DC power output and DC/AC ratios.

            The AC power yield is evaluated using the full pvwatts inverter model in order to account
            for the non-linear behavior of the inverter at low DC power inputs, which is especially
            relevant for low DC/AC ratios and low irradiance conditions. This approach is as using
            `self.yield_ac` but it is more efficient when AC power is computed for several DC/AC ratios
            at once, as `self.yield_ac` would requiere an iteration with repeating calculations every
            loop.

            Parameters
            ----------
            pdc: np.ndarray
                DC power output of the PV system in Watts.

            Uses p_dc_peak, dc_to_ac_ratio, and inverter_effic from the enclosing scope.

            Returns
            -------
            - clipping losses: np.ndarray [Watts]
            - clipped AC power: np.ndarray [Watts]
            """
            # # approximate approach, as in Micheli et al.
            # p_ac_peak = p_dc_peak / dc_to_ac_ratio  # (n_ratios,)
            # pac_unclipped = pdc_centers*inverter_effic  # (n_times or n_bins,)
            # return np.clip(pac_unclipped[:, None] - p_ac_peak[None, :], 0., None)  # (n_times or n_bins, n_ratios)

            p_ac_peak = p_dc_peak / dc_to_ac_ratio  # (n_ratios,) [Watts]
            p_dc0 = p_ac_peak / inverter_effic  # (n_ratios,) [Watts]
            zeta = pdc[:, None] / p_dc0[None, :]  # (n_times or n_bins, n_ratios) [-]
            domain = zeta > 0
            eta = np.zeros_like(zeta, dtype=float)
            eta[domain] = ((inverter_effic/0.9637)
                           *(-0.0162*zeta[domain] - 0.0059/zeta[domain] + 0.9858))  # (n_times or n_bins, n_ratios) [-]
            pac_unclipped = pdc[:, None]*eta  # (n_times or n_bins, n_ratios) [Watts]
            losses = np.clip(pac_unclipped - p_ac_peak[None, :], 0., None)  # (n_times or n_bins, n_ratios) [Watts]
            return losses, np.minimum(pac_unclipped, p_ac_peak[None, :])  # (n_times or n_bins, n_ratios) [Watts]

        if time_series is True:
            logger.debug("Calculating clipping losses time series")
            losses, pac = compute_clipping_losses_numpy(pdc.to_numpy())
            if units == "fraction":
                logger.warning("`units='fraction'` ignored: clipping losses time series are provided in Watts.")
            return pdc.to_frame().assign(
                pac=pac,
                clipping_losses=np.squeeze(losses))

        if method == "explicit":
            logger.debug("Calculating clipping losses using the explicit method")
            losses, _ = compute_clipping_losses_numpy(pdc.to_numpy())  # (n_times, n_ratios) [Watts]
            total_losses = np.nanmean(losses[diurnal], axis=0)  # (n_ratios,) [Watts]

        elif method == "integral":
            logger.debug("Calculating clipping losses using the integral method")
            pdf, bin_edges = np.histogram(pdc.loc[diurnal].dropna(), bins=integral_bins, density=True)  # pdf: (n_bins,) [Watts-1]
            pdc_intervals = np.diff(bin_edges)  # bin intervals: (n_bins,) [Watts]
            pdc_centers = (bin_edges[1:] + bin_edges[:-1]) / 2  # bin centers: (n_bins,) [Watts]
            p_ac_peak = p_dc_peak / dc_to_ac_ratio  # (n_ratios,) [Watts]
            losses, _ = compute_clipping_losses_numpy(pdc_centers)  # (n_bins, n_ratios) [Watts]
            integrand = losses*pdf[:, None]*pdc_intervals[:, None]  # (n_bins, n_ratios) [Watts]
            integrand[pdc_centers[:, None] < p_ac_peak[None, :]] = np.nan  # restricts the integral to the clipping region
            total_losses = np.nansum(integrand, axis=0)  # (n_ratios,) [Watts]

        else:
            raise ValueError(f"unknown method {method=}")

        if units == "fraction":
            total_losses = total_losses / pdc.where(diurnal).mean()  # [-]

        if total_losses.size == 1:
            return total_losses.item()

        return pd.Series(
            data=total_losses,
            index=dc_to_ac_ratio,
            name="clipping_losses",
            dtype=float)

    def optimal_dc_to_ac_ratio(
        self,
        selling_price: float = 40e-6,  # EUR/Wh  (=40 EUR/MWh)
        inverter_cost: float = 0.35,  # EUR/W  (for a 350 kW+ string inverter, typical for commercial PV plants)
        inverter_payback: int = 10,  # years
        clipping_losses_kwargs: dict | None = None,
    ) -> tuple[float, pd.Series]:
        """Calculate the optimal DC/AC ratio of a PV system.

        The optimal DC/AC ratio is the one that maximizes the net savings of the PV system, taking into account
        the cost of the inverter, its payback period, and the selling price of the electricity produced by the
        PV system. The net savings are calculated as the difference between the inverter savings and the cost of
        the inverter prorated over its payback period.

        Parameters
        ----------
        selling_price: float
            Selling price of the electricity produced by the PV system, in EUR/Wh. Default is 40e-6 EUR/Wh (i.e., 40 EUR/MWh).
        inverter_cost: float
            Cost of the inverter, in EUR/W. Default is 0.35 EUR/W, which is typical for a 350 kW+ string inverter used in commercial PV plants.
        inverter_payback: int
            Payback period for the inverter, in years. Default is 10 years.
        clipping_losses_kwargs: dict | None
            Additional kwargs to be passed to the `clipping_losses` method. See `clipping_losses` for details.

        Returns
        -------
        float and pd.Series
            The DC/AC ratio that maximizes the net savings and a pandas Series with the net savings (in EUR/year) for different DC/AC ratios.
        """

        # Análisis aproximado de costos, según una búsqueda en Gemini:
        #   - En términos absolutos, el costo promedio de instalación de una planta FV es de 0.70-0.90 USD/Wp.
        #     Para una planta de 1 MWp: 700-900 kUSD
        #   - En términos relativos, el costo se desglosa en:
        #     - Módulos FV: 35-45% (~320 kUSD)
        #     - Inversor: 8-12% (~80 kUSD)
        #     - Estructura y seguidores: 10-15% (~90 kUSD)
        #     - Cableado, protecciones, transformador: 10-12% (~90 kUSD)
        #     - Instalación y mano de obra: 10-15% (~90 kUSD)
        #     - Otros: 10-15% (~90 kUSD)
        #   - No obstante, en una búsqueda independiente sobre el precio promedio por vatio de un inversor para
        #     planta fotovoltaica he encontrado precios bastante superiores:
        #     - Inversores centrales: son la opción más económica para proyectos 50 MW+, con costos que pueden
        #       bajar hasta 0.10-0.25 USD/W en mercados muy competitivos, como China, o mantenerse cerca de los
        #       0.35 USD/W en otras regiones.
        #     - Inversores de cadena (string): son más caros que los centrales (a veces incluso el doble), pero
        #       los de alta potencia (350 kW+) para aplicaciones comerciales e industriales rondan 0.30-0.40 USD/W
        #     - Con estos datos, y asumiendo un DC/AC ratio de 1.3, el coste del inversor en una planta de 1 MWp
        #       sería de ~230-300 kUSD.

        from scipy.optimize import minimize_scalar

        P_DC_PEAK = 1.  # just a reference value for the calculations [Wp]

        clipping_losses_kwargs = clipping_losses_kwargs or {}
        clipping_losses_kwargs.setdefault("dc_to_ac_ratio", np.linspace(1.0, 1.5, 100))
        clipping_losses_kwargs.setdefault("yield_dc_kwargs", {})
        clipping_losses_kwargs.update({"time_series": False, "units": "W"})
        clipping_losses_kwargs["yield_dc_kwargs"].update({"p_dc_peak": P_DC_PEAK})
        cliploss = self.clipping_losses(**clipping_losses_kwargs) * 8760  # (n_ratios,) [Wh/year]  (8760 h/year)

        dc_to_ac_ratio = cliploss.index
        inverter_savings = (P_DC_PEAK - P_DC_PEAK/dc_to_ac_ratio) * inverter_cost / inverter_payback  # (n_ratios,) [EUR/year]
        selling_loss = cliploss * selling_price  # (n_ratios,) [EUR/year]
        net_savings = inverter_savings - selling_loss  # (n_ratios,) [EUR/year]
        result = minimize_scalar(
            lambda x: -np.interp(x, dc_to_ac_ratio, net_savings),
            bounds=(dc_to_ac_ratio.min(), dc_to_ac_ratio.max()),
            method='bounded')
        net_savings = pd.Series(
            data=net_savings,
            index=dc_to_ac_ratio,
            name="net_savings",
            dtype=float)  # (n_ratios,) [EUR/year]
        return result.x, net_savings

