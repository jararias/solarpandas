# solarpandas: pandas for solar resource assessment

solarpandas is a personal project that I have been developing and using in my own research for years. It integrates under a common framework both standard methods in solar resource modeling and libraries and models that I have developed. The uncomparable extensibility of pandas is the perfect framework for it. The result is an advanced, modern and sophisticated library that combines the unique power and verstility of pandas with the most important methods in solar resource modeling.

## Main features

- solarpandas subclasses the Pandas Series and DataFrame classes adding site location metadata (latitude, longitude, elevation) and free general-purpose custom metadata.

```python
import solarpandas as sp

sdf = sp.SolarSeries(
    data=...,  # as in pandas Series
    index=...,  # a sequence of datetimes, as required by pandas Series
    latitude=37.5,  # mandatory in solarpandas
    longitude=-3.5,  # mandatory in solarpandas
    elevation=900,  # if not providad, set to 0 meters above mean sea level
    custom_metadata={  # optional, free format following json standard rules
        "site": "Jayena",
        "network": "my-network",
        "time_alignment": "center",
    }
)

print(sdf.head())

# metadata has to provided only once. They are automatically 
# propagated afterwards by internal methods in pandas.
```

- It provides fast accessors for key aspects of solar resource modeling, such as calculation of solar position (via [sunwhere](https://github.com/jararias/sunwhere)) and clear-sky irradiance (via [sparta-solar](https://github.com/jararias/sparta-solar)).

```python
sdf = sp.SolarDataFrame(...)
sdf.solpos.zenith  # memory-cached solar zenith angle
sdf.solpos.sunrise(units="utc")  # memory-cached sunrise time, UTC
sdf.lta.ghi  # memory-cached ghi assuming a long-term average clear-sky atmosphere (dni, dif, dir and csi also available)
sdf.cda.ghi  # as the previous but for a clean and dry atmosphere
sdf.clearsky.ghi  # as the previous but using a preset clear-sky atmosphere from those available in sparta-solar
sdf.clearsky.compute(atmosphere="crs_soda", model="SPARTA")  # ad-hoc non-cached calculation of clear sky fluxes

# solar position parameters and clear-sky fluxes are not columns of the dataframe (so it keeps clean and compact) but are always there when needed fast through their corresponding accessors. They act as sort of hidden columns in the dataframe.
```

- solarpandas is shipped with BSRN high-level data retrieval and parsing utilities.

```python
from solarpandas.origin.bsrn import data_availability, load_metadata, load_data

# 1) inspect remote availability (cached locally)
year_table = data_availability(update="auto", as_year_table=True)
print(year_table)

# 2) load station metadata (cached locally)
meta = load_metadata(update="auto")
print(meta.get("car", {}))  # Carpentras example

# 3) load BSRN measurements for one station/year
sdf = load_data(site="car", years=2016, logical_record="LR0100", group="essential")
print(sdf.head())
```

- It has built-in quality-control workflows enhanced with a tailored qc-specific ExtensionDType.

```python
sdf = sp.SolarDataFrame(...)

# 1) perform and show the QC tests results
sdf.qc.tests  # sdf.qc performs and memory-csched the tests, that are accessible through the `tests` dataframe

# 2) access individual tests
sdf.qc.ghi_ppl

# 3) explore individual tests results
sdf.qc.ghi_ppl.counts
sdf.qc.ghi_ppl.pieplot()
sdf.qc.ghi_ppl.heatmap()
sdf.qc.ghi_ppl.plot(sdf)

# 4) bolean masks from sets of individual tests
failed_ghi = sdf.qc.failed(component="ghi")
passed_all = sdf.qc.passed()

# 5) mask failed data points
sdf_masked = sdf.qc.mask_failed(component="ghi")
sdf.qc.heatmap(component="ghi")
```

- It provides specialized plotting helpers for solar datasets through the ``.solarplot`` accessor.

```python
# diurnal line plot
fig1 = sdf.solarplot.diurnal(column="ghi")

# date-time heatmap
fig2 = sdf.solarplot.heatmap(column="ghi", time_ref="tst", twilight_line=True)
```

- as illustrated in the previous examples, solarpandas makes extensive use of disk and memory caching strategies to speed up workflows.
