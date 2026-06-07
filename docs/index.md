
<!-- the caption element below is a workaround to center the image -->
![image title](images/logo_solarpandas_azul_fondo_transparente.png#only-light){ width="400" }
![image title](images/logo_solarpandas_blanco_fondo_transparente.png#only-dark){ width="400" }
/// caption
///


# solarpandas: pandas for solar resource assessment

solarpandas is a pandas-first toolkit for solar irradiance time series.
It extends [pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
and [pandas.Series](https://pandas.pydata.org/docs/reference/api/pandas.Series.html)
with site-location metadata (latitude, longitude, elevation) that travels with the data
through slicing, resampling and I/O, eliminating the need to manage coordinates separately
from measurements.

## What You Can Do

- **Containers** — Use `SolarDataFrame` and `SolarSeries` as drop-in
  replacements for their [pandas](https://pandas.pydata.org/) equivalents;
  metadata is automatically propagated through pandas operations.
- **BSRN data** — Download, cache and load radiation data from any
  [BSRN](https://bsrn.awi.de/) station with a single function call. Supports LR0100
  (basic radiation), LR0300 (net/upwelling) and LR0500 (UV).
- **Solar geometry** — Compute zenith angle, azimuth, cosine of zenith,
  ETR, sunrise/sunset using [sunwhere](https://jararias.github.io/sunwhere)
  via the cached `.solpos` accessor.
- **Clear-sky irradiance** — Estimate clear-sky solar irradiance using
  [sparta-solar](https://jararias.github.io/sparta-solar)
  via the cached `.clearsky` accessor.
- **Quality control** — Provides a set of quality control tests for solar radiation
  data based on [Forstinger et al. (2023)](https://iea-pvps.org/wp-content/uploads/2023/06/IEA-PVPS-T16-05-2023-Worldwide-Benchmark-Modelled-Solar.pdf),
  and references therein, through the `.qc` accessor, which provides a high level
  interface to filter, mask and visualise results by component.
- **Plotting** — Generate diurnal profiles, heatmaps and QC flag visualisations
  tailored to solar data.

## Quick Example of Basic Workflows

```python
from solarpandas.origin import bsrn

# Load data from the Carpentras site of the BSRN network, year 2016
>>> sdf = bsrn.load_data(site="car", years=2016)

# to get solar geometry...
>>> sdf.solpos.zenith
>>> sdf.solpos.azimuth

# to get clear-sky irradiances...
>>> sdf.clearsky.ghi
>>> sdf.clearsky.dif
>>> sdf.lta.ghi  # using a long-term average cloudless atmosphere

# to perform quality control...
>>> sdf.qc.ghi_ppl.flag.counts()  # how many fail and pass the PPL test on GHI
>>> sdf.qc.failed(component="ghi").sum()  # how many fail on all tests for GHI
>>> failed = sdf.qc.failed(component="ghi")
>>> failed.astype(int).solarplot.heatmap(cmap="Reds")  # heatmap of failures over the year

# save to file...
>>> clean = sdf.qc.mask_failed(component="ghi")  # mask all failing GHI instances
>>> clean.to_parquet("clean_ghi.parquet")
```

And `solarpandas` can do any other thing that pandas can do.

## Documentation Structure

- [Installation](installation.md) — setup and environment requirements.
- [User Guide](user-guide.md) — practical workflows and end-to-end examples.
- [Quick References](quick-reference.md) — short command-oriented cheatsheets.
- [API Reference](api.md) — complete API pages generated with mkdocstrings.
