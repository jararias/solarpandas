# Quick References

## Minimal Import

```python
import solarpandas as sp
```

## Containers

```python
import pandas as pd

index = pd.date_range("2016-06-21", periods=1440, freq="1min", tz="UTC")

sdf = sp.SolarDataFrame(
    {"ghi": ..., "dni": ..., "dif": ...},
    index=index,
    latitude=44.083,
    longitude=5.059,
    elevation=100.0,
    custom_metadata={"station": "Carpentras"},
)

ss = sp.SolarSeries(
    ...,
    index=index,
    name="ghi",
    latitude=44.083,
    longitude=5.059,
)
```

## Load Sample Data

```python
sdf = sp.sample_data.load_carpentras_data()
```

## BSRN Data

```python
from solarpandas.origin.bsrn import (
    data_availability,
    load_metadata,
    load_data,
    load_data_from_bsrn_files,
    get_database_path,
    clear_cache,
)

# Availability and metadata
avail = data_availability()                      # dict {site: [files]}
table = data_availability(as_year_table=True)    # compact text table
meta  = load_metadata()                          # dict {site: metadata}

# Load data (high-level, uses Parquet cache)
sdf = load_data(site="car", years=2016)
sdf = load_data(site="car", years=[2015, 2016, 2017])
sdf = load_data(site="car", years=2016, logical_record="LR0300")
sdf = load_data(site="car", years=2016, group="all")

# Low-level FTP access
sdf = load_data_from_bsrn_files(site="car", years=2016)
sdf, meta_df = load_data_from_bsrn_files(site="car", years=2016, include_metadata=True)
sdf, extra   = load_data_from_bsrn_files(site="car", years=2016, extra_records=["LR0300"])

# Cache management
print(get_database_path())
clear_cache(site="car", year=2016, logical_record="LR0100")
clear_cache()   # clear everything
```

## Read and Write

```python
sdf.to_parquet("data.parquet")
sdf.to_csv("data.csv")

sdf = sp.read_parquet("data.parquet")
sdf = sp.read_csv("data.csv")
```

## Solar Position

```python
sdf.solpos.zenith
sdf.solpos.azimuth
sdf.solpos.cosz
sdf.solpos.etn
sdf.solpos.sunrise(units="utc")
sdf.solpos.sunset(units="utc")
```

## Clear-Sky

```python
sdf.clearsky.ghi   # clear-sky GHI [W m⁻²]
sdf.clearsky.dni   # clear-sky DNI [W m⁻²]
sdf.clearsky.dif   # clear-sky DIF [W m⁻²]
sdf.clearsky.csi   # clear-sky circumsolar irradiance [W m⁻²]
```

## Quality Control

```python
# All test flags as a DataFrame (+1 passes / -1 fails / 0 not verifiable)
sdf.qc.tests

# Access one test
sdf.qc["ghi_ppl"]
sdf.qc.ghi_ppl

# Filter tests
sdf.qc.filter(component="ghi")              # fixed component mapping
sdf.qc.filter(tests=["ghi_ppl", "closure"]) # explicit list
sdf.qc.filter(like="erl")                   # substring
sdf.qc.filter(regex=r"^K")                  # regex

# Boolean masks
sdf.qc.failed(component="ghi")   # True where ≥1 test fails
sdf.qc.passed(component="ghi")   # True where all tests pass or N/A

# Mask bad values (NaN by default)
sdf.qc.mask_failed(component="ghi")
sdf.qc.mask_failed(tests=["closure"])

# QC heatmap
sdf.qc.heatmap(component="ghi")
sdf.qc.heatmap(tests=["ghi_ppl", "ghi_erl"])
```

### Available QC test names

| Name | Type | Needs |
|------|------|-------|
| `ghi_ppl` | PPL | ghi |
| `dif_ppl` | PPL | dif |
| `dni_ppl` | PPL | dni |
| `ghi_erl` | ERL | ghi |
| `dif_erl` | ERL | dif |
| `dni_erl` | ERL | dni |
| `Kn_ppl` | K-space | ghi, dni |
| `Kn_erl` | K-space | ghi, dni |
| `KT_erl` | K-space | ghi |
| `K_erl` | K-space | ghi, dni |
| `K_erl_clear` | K-space | ghi, dni |
| `closure` | Closure | ghi, dni, dif |
| `trackeroff` | Tracker | ghi, dni |

## Plotting

```python
sdf.solarplot.diurnal(column="ghi")
sdf.solarplot.heatmap(column="ghi", time_ref="tst", twilight_line=True)
sdf.solarplot.rolling(column="ghi", window_size=3, max_sza=95.0)
sdf.qc.heatmap(component="ghi")
```

## Cache Helpers

```python
import solarpandas as sp

sp.get_solpos_cache_info()
sp.clear_solpos_cache()

sp.get_clearsky_cache_info()
sp.clear_clearsky_cache()

sp.get_qc_cache_info()
sp.clear_qc_cache()
```
