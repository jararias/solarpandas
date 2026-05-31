# solarpandas

solarpandas is a pandas-first toolkit for solar irradiance time series.

It provides:

- Metadata-aware containers with site information.
- Accessors for solar position and clear-sky irradiance.
- BSRN retrieval and parsing utilities.
- Quality-control workflows for irradiance data.
- Plotting helpers specialized for solar datasets.

## What You Can Do

- Work with SolarDataFrame and SolarSeries while preserving metadata.
- Compute solar geometry and derived quantities with caching.
- Estimate clear-sky irradiance components.
- Run qcrad-style quality-control checks.
- Explore station availability and metadata from BSRN sources.

## Quick Example

```python
import solarpandas as sp

sdf = sp.sample_data.load_carpentras_data()
print(sdf.head())
print(sdf.solpos.zenith.head())
print(sdf.clearsky.ghi.head())
```

## Documentation Structure

- Installation: setup and environment requirements.
- User Guide: practical workflows and end-to-end examples.
- Quick References: short command-oriented cheatsheets.
- API Reference: complete API pages generated with mkdocstrings.
