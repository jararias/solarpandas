# solarpandas: pandas for solar resource assessment

solarpandas is a personal project that integrates under a common framework multiple models and libraries that I have developed throughout my research career and standard methods in solar resource modeling.

## Main features

- solarpandas subclasses Pandas Series and DataFrame classes adding site location metadata (latitude, longitude, elevation) and free general-purpose custom metadata.

- It provides fast accessors for key aspects of solar resource modeling, such as calculation of solar position (via [sunwhere](https://github.com/jararias/sunwhere)) and clear-sky irradiance (via [sparta-solar](https://github.com/jararias/sparta-solar)).

- solarpandas is shipped with BSRN high-level data retrieval and parsing utilities.

- It has built-in quality-control workflows enhanced with a tailored qc-specific ExtensionDType.

- It provides specialized plotting helpers for solar datasets through the ``.solarplot`` accessor.

- solarpandas makes extensive use of disk and memory caching strategies to speed up workflows.

## Quick Snippets

```python
import solarpandas as sp
```

### BSRN: Availability, Metadata, and Data Loading

```python
from solarpandas.origin.bsrn import data_availability, load_metadata, load_data

# 1) Inspect remote availability (cached locally)
year_table = data_availability(update="auto", as_year_table=True)
print(year_table)

# 2) Load station metadata (cached locally)
meta = load_metadata(update="auto")
print(meta.get("car", {}))  # Carpentras example

# 3) Load BSRN measurements for one station/year
sdf = load_data(site="car", years=2016, logical_record="LR0100", group="essential")
print(sdf.head())
```

## Solar Position

```python
# Solar position accessor (cached)
zenith = sdf.solpos.zenith
azimuth = sdf.solpos.azimuth
sunrise_utc = sdf.solpos.sunrise(units="utc")

print(zenith.head())
print(sunrise_utc.head())
```

## Clear-Sky Irradiance

```python
# Clear-sky accessor (cached)
ghi_cs = sdf.clearsky.ghi
dni_cs = sdf.clearsky.dni

# One-off computation without using cache
cs = sdf.clearsky.compute(atmosphere="crs_soda", model="SPARTA")
print(ghi_cs.head())
```

## Quality Control

```python
# Run all QC tests
qc = sdf.qc

# Get raw test dataframe
tests = qc.tests

# Boolean masks
failed_ghi = qc.failed(component="ghi")
passed_all = qc.passed()

# Mask failed data points
sdf_masked = qc.mask_failed(component="ghi")

print(tests.columns)
print(failed_ghi.sum(), passed_all.sum())
```

## solarplot Examples

```python
# Diurnal line plot for selected variable(s)
fig1 = sdf.solarplot.diurnal(column="ghi")

# Date-time heatmap
fig2 = sdf.solarplot.heatmap(column="ghi", time_ref="tst", twilight_line=True)

# QC heatmap
fig3 = sdf.qc.heatmap(component="ghi")
```

## Included Sample Data

```python
sdf = sp.sample_data.load_carpentras_data()
print(sdf.index.min(), sdf.index.max())
```

## Notes

- Most accessors expect a SolarDataFrame/SolarSeries with a DateTimeIndex.
- Several computations assume 1-minute data for best QC behavior.
- Cache helper functions are available at package level:
  - sp.get_solpos_cache_info(), sp.clear_solpos_cache()
  - sp.get_clearsky_cache_info(), sp.clear_clearsky_cache()
  - sp.get_qc_cache_info(), sp.clear_qc_cache()
