# User Guide

## 1. Load Data

### Sample Dataset

```python
import solarpandas as sp

sdf = sp.sample_data.load_carpentras_data()
```

### BSRN Dataset

```python
from solarpandas.origin.bsrn import data_availability, load_metadata, load_data

availability = data_availability(update="auto")
metadata = load_metadata(update="auto")
sdf = load_data(site="car", years=2016, logical_record="LR0100", group="essential")
```

## 2. Solar Position

```python
zenith = sdf.solpos.zenith
azimuth = sdf.solpos.azimuth
sunrise = sdf.solpos.sunrise(units="utc")
```

## 3. Clear-Sky Irradiance

```python
ghi_cs = sdf.clearsky.ghi
dni_cs = sdf.clearsky.dni
dif_cs = sdf.clearsky.dif
```

## 4. Quality Control

```python
qc = sdf.qc
tests = qc.tests
failed_ghi = qc.failed(component="ghi")
masked = qc.mask_failed(component="ghi")
```

## 5. Plotting

```python
fig1 = sdf.solarplot.diurnal(column="ghi")
fig2 = sdf.solarplot.heatmap(column="ghi", time_ref="tst", twilight_line=True)
fig3 = sdf.qc.heatmap(component="ghi")
```

## 6. Persist Data

```python
sdf.to_parquet("dataset.parquet")
sdf_reloaded = sp.read_parquet("dataset.parquet")
```

## Best Practices

- Keep a consistent DateTimeIndex with explicit timezone handling.
- Check missing values before running quality-control tests.
- Reuse cached accessors for repeated computations.
