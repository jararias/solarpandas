# solarpandas

solarpandas is a pandas-first toolkit for solar irradiance time series.
It provides:

- Data containers with site metadata (latitude, longitude, elevation)
- Fast accessors for solar position and clear-sky irradiance
- BSRN data retrieval and parsing utilities
- Built-in quality-control workflows (qcrad-style checks)
- Plotting helpers specialized for solar datasets

## Main Features

- Metadata-aware containers:
  - SolarSeries
  - SolarDataFrame
- Solar position via sunwhere:
  - zenith, azimuth, sunrise/sunset, true solar time, and more
- Clear-sky irradiance via spartasolar:
  - ghi, dni, dif, csi
- BSRN support:
  - inspect remote availability
  - load station metadata
  - load and cache measurement records
- Quality control:
  - physically possible and extremely rare limits
  - closure and tracker-related checks
- Solar plots:
  - diurnal views
  - date-time heatmaps

## Quick Start

```python
import solarpandas as sp
```

## BSRN: Availability, Metadata, and Data Loading

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
