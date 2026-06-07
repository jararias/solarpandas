# User Guide

This guide walks you through the main workflows in solarpandas: creating
metadata-aware containers, downloading BSRN station data, computing solar
geometry and clear-sky irradiance, running quality-control tests, and
visualizing results.

---

## 1. Containers: SolarDataFrame and SolarSeries

All data in solarpandas is stored in `SolarDataFrame` or `SolarSeries`
objects. These are direct subclasses of `pandas.DataFrame` and
`pandas.Series` that attach site location metadata — latitude, longitude,
elevation — to the data object itself, so that metadata is never out of
sync with the measurements.

### Creating a SolarDataFrame from scratch

```python
import pandas as pd
import solarpandas as sp

index = pd.date_range("2016-06-21", periods=1440, freq="1min", tz="UTC")

sdf = sp.SolarDataFrame(
    {"ghi": ..., "dni": ..., "dif": ...},
    index=index,
    latitude=44.083,
    longitude=5.059,
    elevation=100.0,
)
```

`latitude` and `longitude` are required and must be provided. `elevation`
is optional and defaults to `0.0` (metres above mean sea level) if omitted.

### Custom metadata

Any additional site information (station name, network, instrument serial
numbers, etc.) can be stored in a free-form dictionary under
`custom_metadata`:

```python
sdf = sp.SolarDataFrame(
    ...,
    latitude=44.083,
    longitude=5.059,
    custom_metadata={
        "station": "Carpentras",
        "network": "BSRN",
        "station_id": "CAR",
    },
)

print(sdf.custom_metadata["station"])  # Carpentras
```

### Metadata propagation

Metadata are propagated automatically through most pandas operations:

```python
subset = sdf["ghi"]           # SolarSeries — keeps lat/lon/elv
day = sdf.loc["2016-06-21"]   # SolarDataFrame — keeps metadata
resampled = sdf.resample("1h").mean()  # metadata preserved
```

---

## 2. Loading Data

### Sample dataset

The package ships a bundled one-year dataset (Carpentras BSRN station, 2016)
suitable for offline exploration:

```python
import solarpandas as sp

sdf = sp.sample_data.load_carpentras_data()
print(sdf.shape)         # (527040, 3) — one year at 1-minute resolution (2016 is a leap year)
print(sdf.columns)       # Index(['ghi', 'dni', 'dif'], ...)
print(sdf.latitude)      # 44.083
```

### Reading from disk

solarpandas can round-trip data through Parquet and CSV while preserving
all metadata:

```python
# Write
sdf.to_parquet("dataset.parquet")
sdf.to_csv("dataset.csv")

# Read back (metadata is restored automatically)
sdf2 = sp.read_parquet("dataset.parquet")
sdf3 = sp.read_csv("dataset.csv")
```

---

## 3. BSRN Data

The `solarpandas.origin.bsrn` module provides a full pipeline to download,
cache and load data from the
[Baseline Surface Radiation Network](https://www.pangaea.de/search/?q=BSRN).
All network I/O is funnelled through a local cache directory so that
subsequent calls are instant.

### 3.1 Checking data availability

Before downloading, it is useful to know which stations and years are
present on the FTP server:

```python
from solarpandas.origin.bsrn import data_availability

# Returns a dict {site: [list of monthly filenames on the FTP server]}
availability = data_availability()

# List sites with at least one file
print(list(availability.keys()))

# Pretty-print a compact year × site table (one char per year, # = has data)
table = data_availability(as_year_table=True)
print(table)
```

The `update` parameter controls when the local cache is refreshed:

| Value | Behaviour |
|-------|-----------|
| `"auto"` (default) | Refresh if the local cache is older than 7 days |
| `True` | Always refresh from the FTP server |
| `False` | Never refresh; read the local cache even if stale |

You can also save the availability table to a file for later reference:

```python
data_availability(year_table_output="availability.txt")
```

### 3.2 Station metadata

Full station metadata (coordinates, altitude, surface type, etc.) can be
loaded from PANGAEA:

```python
from solarpandas.origin.bsrn import load_metadata

meta = load_metadata()          # refresh automatically when > 7 days old
meta_car = meta["car"]          # dict with all fields for Carpentras

print(meta_car["latitude"])
print(meta_car["longitude"])
print(meta_car["elevation"])
```

The same `update` semantics as `data_availability` apply.

### 3.3 Loading radiation data

`load_data` is the primary function for retrieving BSRN data. It builds
and maintains a local Parquet cache so that data is parsed from the raw
`.dat.gz` FTP files only once per site/year combination.

#### Basic usage — one year, one station

```python
from solarpandas.origin.bsrn import load_data

sdf = load_data(site="car", years=2016)
print(sdf.columns)   # ['ghi', 'dni', 'dif', ...] — essential group
```

#### Multiple years

```python
sdf = load_data(site="car", years=[2015, 2016, 2017])
print(sdf.index[[0, -1]])  # spans three years
```

#### Logical records

The BSRN file format organises measurements into numbered *logical records*
(LR). The most commonly used are:

| Logical Record | Content |
|---------------|---------|
| `LR0100` (default) | Basic radiation: GHI, DNI, DIF, LWD, net, temperature |
| `LR0300` | Other radiation: net radiation, upward shortwave and longwave |
| `LR0500` | UV irradiance |

```python
# Downwelling longwave and net radiation
sdf_net = load_data(site="car", years=2016, logical_record="LR0300")
print(sdf_net.columns)

# UV irradiance
sdf_uv = load_data(site="car", years=2016, logical_record="LR0500")
```

#### Variable groups

Within a logical record, `group` selects a subset of variables based on
CF-metadata tags:

| Group | Variables included |
|-------|--------------------|
| `"essential"` (default) | The main radiation components (GHI, DNI, DIF, …) |
| `"avg"` | Averages — removes instantaneous ancillary channels |
| `"all"` | Every column present in the raw file |

```python
sdf_all = load_data(site="car", years=2016, group="all")
print(sdf_all.columns)  # includes ancillary columns like temperatures
```

### 3.4 Low-level access: loading directly from FTP files

`load_data_from_bsrn_files` gives finer control. It reads raw `.dat.gz`
files (downloading them on demand) and returns data in original BSRN column
names without the Parquet caching layer:

```python
from solarpandas.origin.bsrn import load_data_from_bsrn_files

# Case 1 — radiation data only
sdf = load_data_from_bsrn_files(site="car", years=2016)

# Case 2 — radiation data + station metadata records
sdf, meta_df = load_data_from_bsrn_files(
    site="car", years=2016, include_metadata=True
)

# Case 3 — radiation data + an extra logical record
sdf, extra = load_data_from_bsrn_files(
    site="car", years=2016, extra_records=["LR0300"]
)
sdf_lr300 = extra["LR0300"]

# Load only a specific month
sdf_june = load_data_from_bsrn_files(site="car", years=2016, months=6)
```

### 3.5 Cache management

The local Parquet cache lives under the platform-specific user data
directory (typically `~/.local/share/solarpandas/bsrn/cached/`).

```python
from solarpandas.origin.bsrn import get_database_path, clear_cache

# Inspect the cache location
print(get_database_path())

# Clear one specific year/site/record combination
clear_cache(site="car", year=2016, logical_record="LR0100")

# Clear an entire site
clear_cache(site="car")

# Clear everything (omit all arguments)
clear_cache()
```

Clearing the cache forces a fresh download and re-parse on the next call
to `load_data`.

---

## 4. Solar Position

The `.solpos` accessor computes solar geometry on demand using the
[sunwhere](https://github.com/jararias/sunwhere) library. Results are
cached in memory so repeated access is free.

```python
sdf = sp.sample_data.load_carpentras_data()

zenith   = sdf.solpos.zenith    # solar zenith angle [degrees]
azimuth  = sdf.solpos.azimuth   # solar azimuth angle [degrees from N]
cosz     = sdf.solpos.cosz      # cosine of zenith angle
etn      = sdf.solpos.etn       # extraterrestrial normal irradiance [W m⁻²]

sunrise  = sdf.solpos.sunrise(units="utc")   # sunrise times
sunset   = sdf.solpos.sunset(units="utc")    # sunset times
```

The solar-position algorithm and refraction correction can be configured
globally through `sp.set_option`:

```python
sp.set_option("solpos.algorithm", "psa")
sp.set_option("solpos.refraction", True)
```

After changing options, clear the cache to force recomputation:

```python
sp.clear_solpos_cache()
```

---

## 5. Clear-Sky Irradiance

The `.clearsky` accessor provides modelled clear-sky GHI, DNI and DIF.
Like solar position, results are cached automatically.

```python
ghi_cs = sdf.clearsky.ghi   # clear-sky global horizontal [W m⁻²]
dni_cs = sdf.clearsky.dni   # clear-sky direct normal [W m⁻²]
dif_cs = sdf.clearsky.dif   # clear-sky diffuse horizontal [W m⁻²]
csi    = sdf.clearsky.csi   # clear-sky circumsolar irradiance [W m⁻²]
```

Clear-sky estimates are used internally by several quality-control tests
that compare measured values against modelled thresholds under cloud-free
conditions.

---

## 6. Quality Control

solarpandas implements the
[QCRAD](https://doi.org/10.1175/JTECH-D-21-0094.1) quality-control
methodology through the `.qc` accessor. Tests are run lazily the first
time `.qc` is accessed and cached in memory for the lifetime of the
object. Every test assigns one of three flag values to each timestamp:

| Flag | Integer value | Meaning |
|------|:---:|---------|
| `passes` | **+1** | The measurement satisfies the test condition |
| `fails` | **-1** | The measurement violates the test condition |
| `not_verifiable` | **0** | The test cannot be applied (missing column, sun below horizon, etc.) |

### 6.1 Available tests

Tests are grouped by the type of check they perform:

#### Physically Possible Limits (PPL)

Compare each component against theoretical upper and lower bounds derived
from the extraterrestrial irradiance and solar zenith angle. A measurement
outside these bounds is considered physically impossible.

| Test | Component checked |
|------|------------------|
| `ghi_ppl` | Global Horizontal Irradiance |
| `dif_ppl` | Diffuse Horizontal Irradiance |
| `dni_ppl` | Direct Normal Irradiance |

#### Extremely Rare Limits (ERL)

Tighter bounds than PPL. Failing an ERL test means the value is
theoretically possible but statistically very rare. Simultaneous failure
of both PPL and ERL indicates likely sensor error.

| Test | Component checked |
|------|------------------|
| `ghi_erl` | Global Horizontal Irradiance |
| `dif_erl` | Diffuse Horizontal Irradiance |
| `dni_erl` | Direct Normal Irradiance |

#### K-space consistency tests

These tests operate in the dimensionless clearness-index space ($K_T$, $K_n$,
$K$) where systematic instrument errors create characteristic signatures.

| Test | Description |
|------|-------------|
| `Kn_ppl` | Direct-beam clearness index $K_n$ must not exceed the global clearness index $K_T$ |
| `Kn_erl` | $K_n$ against extremely-rare upper limits |
| `KT_erl` | $K_T$ against extremely-rare upper limits |
| `K_erl` | Combined $K$-index against extremely-rare limits |
| `K_erl_clear` | Same as `K_erl` but restricted to clear-sky conditions |

#### Radiative closure

The closure test verifies the physical identity
$\text{GHI} \approx \text{DNI} \cdot \cos\theta_z + \text{DIF}$.
Significant deviations indicate a miscalibrated instrument, a misaligned
tracker, or shading errors. The tolerance is relaxed at high zenith angles
(above 75°).

| Test | Description |
|------|-------------|
| `closure` | GHI = DNI·cos(θ) + DIF consistency, requires all three components |

#### Tracker status

The tracker test detects likely tracker-off events by comparing measured
DNI and GHI against clear-sky model values. When the tracker is off, DNI
drops sharply while GHI remains close to modelled levels.

| Test | Description |
|------|-------------|
| `trackeroff` | Detects likely tracker misalignment or off-pointing |

### 6.2 Accessing raw test results

`.qc.tests` returns a `DataFrame` where each column is one test and each
row is a timestamp. Values are `+1`, `-1` or `0` as described above.

```python
sdf = sp.sample_data.load_carpentras_data()

tests = sdf.qc.tests
print(tests.columns.tolist())
# ['ghi_ppl', 'dif_ppl', 'dni_ppl', 'ghi_erl', 'dif_erl', 'dni_erl',
#  'Kn_ppl', 'Kn_erl', 'KT_erl', 'K_erl', 'K_erl_clear',
#  'closure', 'trackeroff']

# Access a single test as a Series
ghi_ppl_flags = sdf.qc["ghi_ppl"]
# or equivalently:
ghi_ppl_flags = sdf.qc.ghi_ppl
```

### 6.3 Filtering tests

`.qc.filter` returns a sub-DataFrame of tests matching a component,
explicit names, or a pattern:

```python
# All tests that involve GHI
ghi_tests = sdf.qc.filter(component="ghi")

# Select tests by explicit list
subset = sdf.qc.filter(tests=["ghi_ppl", "ghi_erl", "closure"])

# Select tests whose name contains "erl"
erl_tests = sdf.qc.filter(like="erl")

# Select tests matching a regex
k_tests = sdf.qc.filter(regex=r"^K")
```

The `component` shortcut uses a fixed mapping:

- `"ghi"` → PPL, ERL, K-space, closure, tracker tests that involve GHI
- `"dni"` → PPL, ERL, K-space, closure, tracker tests that involve DNI
- `"dif"` → PPL, ERL and closure tests for DIF

### 6.4 Creating boolean masks

```python
# Boolean Series: True where at least one GHI-related test fails
ghi_bad = sdf.qc.failed(component="ghi")
print(ghi_bad.sum(), "flagged timestamps")

# Boolean Series: True where all selected tests pass or are not verifiable
ghi_good = sdf.qc.passed(component="ghi")

# Work with an explicit set of tests instead of a component
bad_ppl = sdf.qc.failed(tests=["ghi_ppl", "dni_ppl", "dif_ppl"])
```

### 6.5 Masking bad data

`.qc.mask_failed` replaces values at flagged timestamps with `NaN` (or
any fill value) while keeping the rest of the dataframe intact.

```python
# Mask only the 'ghi' column where any GHI test fails
clean = sdf.qc.mask_failed(component="ghi")
print(clean["ghi"].isna().sum(), "values masked")

# Mask only where the closure test fails (affects all three components)
clean_closure = sdf.qc.mask_failed(tests=["closure"])

# Mask with a custom fill value instead of NaN
clean_fill = sdf.qc.mask_failed(component="ghi", other=-9999.0)
```

### 6.6 Heatmap visualisation

`.qc.heatmap` renders a pass/fail heatmap over the time axis, one row per
test, which makes it easy to spot systematic patterns (e.g., a tracker that
goes off every afternoon):

```python
# Heatmap for all tests related to DNI
fig = sdf.qc.heatmap(component="dni")

# Heatmap for a custom test subset
fig = sdf.qc.heatmap(tests=["ghi_ppl", "ghi_erl", "closure"])

# Combined heatmap: encodes failure severity by component group
fig = sdf.qc.heatmap(component="ghi", combined=True)
```

### 6.7 Typical quality-control workflow

```python
import solarpandas as sp

# 1. Load one year of Carpentras data
sdf = sp.sample_data.load_carpentras_data().loc["2016"]

# 2. Inspect which fraction of timestamps fail each component
for comp in ("ghi", "dni", "dif"):
    n_fail = sdf.qc.failed(component=comp).sum()
    pct = 100 * n_fail / len(sdf)
    print(f"{comp}: {n_fail} flagged ({pct:.2f}%)")

# 3. Produce a clean dataset by masking all three components
clean_sdf = sdf.copy()
for comp in ("ghi", "dni", "dif"):
    clean_sdf = clean_sdf.qc.mask_failed(component=comp)

# 4. Inspect the closure test separately
closure_fails = sdf.qc.failed(tests=["closure"])
print(f"closure failures: {closure_fails.sum()}")

# 5. Visualise the flagging pattern for GHI
fig = sdf.qc.heatmap(component="ghi")
fig.savefig("ghi_qc_heatmap.png", dpi=150)
```

### 6.8 Cache and performance

QC results are memoised by dataframe content. Accessing `.qc` on the same
object multiple times costs nothing after the first call:

```python
import solarpandas as sp

sp.get_qc_cache_info()   # {'hits': 0, 'misses': 0, ...}
_ = sdf.qc.tests         # triggers computation
sp.get_qc_cache_info()   # {'hits': 0, 'misses': 1, ...}
_ = sdf.qc.tests         # served from cache
sp.get_qc_cache_info()   # {'hits': 1, 'misses': 1, ...}

# Force recomputation (e.g., after modifying the data)
sp.clear_qc_cache()
```

---

## 7. Plotting

The `.solarplot` accessor provides solar-specific plot types. All functions
return a `matplotlib.figure.Figure` that can be customised or saved normally.

```python
# Diurnal profile on a compressed daytime-only timeline
fig = sdf.solarplot.diurnal(column="ghi")

# Julian-day × time-of-day heatmap
fig = sdf.solarplot.heatmap(column="ghi", time_ref="tst", twilight_line=True)

# Heatmap in UTC (useful to spot clock offsets)
fig = sdf.solarplot.heatmap(column="ghi", time_ref="utc")

# Interactive day-by-day rolling plot (scroll or arrow keys to navigate)
fig = sdf.solarplot.rolling(column="ghi", window_size=3, max_sza=95.0)
fig = sdf.solarplot.rolling(column=["ghi", "dni"], y_scale="global")

# QC heatmap (test flags over time, see Section 6.6)
fig = sdf.qc.heatmap(component="ghi")
```

---

## 8. Persist Data

```python
# Write to Parquet (recommended: compact and lossless)
sdf.to_parquet("dataset.parquet")
sdf_reloaded = sp.read_parquet("dataset.parquet")

# Write to CSV (portable but larger)
sdf.to_csv("dataset.csv")
sdf_reloaded = sp.read_csv("dataset.csv")
```

Both readers restore latitude, longitude, elevation and `custom_metadata`
from embedded file metadata automatically.

---

## 9. Cache Utilities

solarpandas maintains three independent in-memory caches: solar position,
clear-sky irradiance and quality control. Each can be inspected and cleared
independently:

```python
import solarpandas as sp

# Solar position cache
sp.get_solpos_cache_info()
sp.clear_solpos_cache()

# Clear-sky irradiance cache
sp.get_clearsky_cache_info()
sp.clear_clearsky_cache()

# Quality-control cache
sp.get_qc_cache_info()
sp.clear_qc_cache()
```

Caches survive as long as the Python session is alive. Call the clear
functions after modifying data in-place or after changing global options.

---

## Best Practices

- Always attach an explicit timezone to the `DatetimeIndex` (use `tz="UTC"`
  or localise before passing to solarpandas).
- Check `sdf.isna().sum()` before running QC — missing values are treated
  as `not_verifiable` by all tests.
- Prefer `load_data` over `load_data_from_bsrn_files` for day-to-day work;
  the cached Parquet layer makes repeated runs much faster.
- After modifying a dataframe in-place, clear the QC and clear-sky caches
  to ensure stale results are not served.
- Use `group="essential"` (the default) unless you specifically need
  ancillary channels — it keeps the dataframe compact and readable.
