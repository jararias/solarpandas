
<!-- the caption element below is a workaround to center the image -->
![image title](images/logo_solarpandas_fondo_blanco.png#only-light){ width="400" }
![image title](images/logo_solarpandas_fondo_negro.png#only-dark){ width="400" }
/// caption
///

# solarpandas: pandas for solar resource assessment

solarpandas is a pandas-first toolkit for solar irradiance time series.
It extends `pandas.DataFrame` and `pandas.Series` with site-location
metadata (latitude, longitude, elevation) that travels with the data
through slicing, resampling and I/O, eliminating the need to manage
coordinates separately from measurements.

## What You Can Do

- **Containers** — Use `SolarDataFrame` and `SolarSeries` as drop-in
  replacements for their pandas equivalents; metadata is automatically
  propagated through pandas operations.
- **BSRN data** — Download, cache and load radiation data from any
  BSRN station with a single function call. Supports LR0100
  (basic radiation), LR0300 (net/upwelling) and LR0500 (UV).
- **Solar geometry** — Compute zenith angle, azimuth, cosine of zenith,
  ETR, sunrise/sunset via the cached `.solpos` accessor.
- **Clear-sky irradiance** — Estimate GHI, DNI and DIF under cloud-free
  conditions via the cached `.clearsky` accessor.
- **Quality control** — Run the full QCRAD test battery (PPL, ERL,
  K-space, closure, tracker) through the `.qc` accessor. Filter,
  mask and visualise results by component.
- **Plotting** — Generate diurnal profiles, heatmaps and QC flag
  visualisations tailored to solar data.

## Quick Example

```python
import solarpandas as sp

# Load the bundled sample dataset (Carpentras BSRN, 2016, 1-minute)
sdf = sp.sample_data.load_carpentras_data()

# Solar geometry
print(sdf.solpos.zenith.describe())

# Clear-sky irradiance
print(sdf.clearsky.ghi.describe())

# Quality control — how many GHI timestamps fail any test?
n_bad = sdf.qc.failed(component="ghi").sum()
print(f"Flagged GHI timestamps: {n_bad}")

# Mask bad values and save
clean = sdf.qc.mask_failed(component="ghi")
clean.to_parquet("clean_ghi.parquet")
```

## Documentation Structure

- [Installation](installation.md) — setup and environment requirements.
- [User Guide](user-guide.md) — practical workflows and end-to-end examples.
- [Quick References](quick-references.md) — short command-oriented cheatsheets.
- [API Reference](api.md) — complete API pages generated with mkdocstrings.
