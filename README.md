
<p align="center">
<img src="https://raw.githubusercontent.com/jararias/solarpandas/main/docs/images/logo_solarpandas_azul_fondo_transparente.png" alt="logo" width="50%">
</p>

# solarpandas: pandas for solar resource assessment

![python versions](https://img.shields.io/badge/python-3.13-blue.svg)
![tests-badge](https://raw.githubusercontent.com/jararias/solarpandas/main/docs/images/tests-badge.svg)
![coverage-badge](https://raw.githubusercontent.com/solarpandas/sunwhere/main/docs/images/coverage-badge.svg)
[![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-green.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

𝘴𝘰𝘭𝘢𝘳𝘱𝘢𝘯𝘥𝘢𝘴 is a personal project that I have been developing and using for my own research for years. It integrates under a common framework both standard methods in solar resource modeling and libraries and models I have developed myself. The incomparable extensibility of pandas makes it the perfect framework for this. The result is an advanced, modern, and sophisticated library that combines the unique power and versatility of pandas with the most widely used methods in solar resource modeling.

## Main features

- **𝘴𝘰𝘭𝘢𝘳𝘱𝘢𝘯𝘥𝘢𝘴 subclasses pandas Series and DataFrame** to embed site location metadata (latitude, longitude and elevation) and optional general-purpose custom metadata. This approach frees the user from having to pass location metadata to every individual routine, as it is automatically propagated across objects and only needs to be specified once, while retaining the rich API of pandas in the SolarSeries and SolarDataFrame objects.

```python
>>> import solarpandas as sp

#  pandas class   solarpandas class
#  ------------   -----------------
#   Series         SolarSeries
#   DataFrame      SolarDataFrame

>>> sdf = sp.SolarSeries(
...     data=np.linspace(500, 550, 6),  # as in pandas Series
...     index=pd.date_range("2026-06-01 10", periods=6, freq="30min"),  # a sequence of datetimes, as required by pandas Series
...     name="ghi",
...     # metadata...
...     latitude=36.949,  # mandatory in solarpandas
...     longitude=-3.823,  # mandatory in solarpandas
...     elevation=914,  # if not providad, set to 0 meters above mean sea level
...     custom_metadata={  # optional, following json standard rules
...         "site": "Jayena",
...         "network": "my-network",
...     }
... )
>>> sdf
2026-06-01 10:00:00    500.0
2026-06-01 10:30:00    510.0
2026-06-01 11:00:00    520.0
2026-06-01 11:30:00    530.0
2026-06-01 12:00:00    540.0
2026-06-01 12:30:00    550.0
Freq: 30min, Name: ghi, dtype: float64
[site=Jayena/my-network latitude=36.9490° longitude=-3.8230° elevation=914.0 m]

>>> sdf_hourly = sdf.resample("h").mean()
>>> sdf_hourly
2026-06-01 10:00:00    505.0
2026-06-01 11:00:00    525.0
2026-06-01 12:00:00    545.0
Freq: h, dtype: float64
[site=Jayena/my-network latitude=36.9490° longitude=-3.8230° elevation=914.0 m]
```

- SolarDataFrame instances **can be serialized and de-serialized** to and from `parquet` or `csv` files **keeping the original metadata**. This opens the door to standardized metadata for solar time series following cf-compliant rules.

```python
>>> sdf = sp.sample_data.load_carpentras_data()
>>> sdf.custom_metadata
{'station': 'CAR',
 'location': 'Carpentras, France',
 'network': 'BSRN',
 'source': 'BSRN FTP server via solarpandas',
 'institution': 'Jose A Ruiz-Arias (solarpandas dev) and BSRN data providers',
 'contact': 'xxx@xxx.xxx',
 'timestamp_alignment': 'center',
 'surface_type': 'cultivated',
 'topography_type': 'hilly, rural',
  ...
 'variables': {
  'ghi': {
    'standard_name': 'surface_downwelling_shortwave_flux_in_air',
    'long_name': 'global horizontal irradiance',
    'short_name': 'ghi',
    'units': 'W m-2',
    'cell_methods': 'time: mean (interval: 1 minute)',
    'bsrn_name': 'global_horizontal_avg'
  },
  ...
 }
}
```

- 𝘴𝘰𝘭𝘢𝘳𝘱𝘢𝘯𝘥𝘢𝘴 provides **fast memory-cached accessors for** key aspects of solar resource modeling, such as the calculation of **solar position** (via [sunwhere](https://github.com/jararias/sunwhere)) **and clear-sky irradiance** (via [sparta-solar](https://github.com/jararias/sparta-solar)). These parameters are not stored as columns of the dataframe, keeping it clean and compact, but are instead exposed as virtual columns through the accessors.

```python
>>> sdf = sp.sample_data.load_carpentras_data()
>>> sdf
                           ghi  dni  dif
time                                    
2016-01-01 00:00:30+00:00 -1.0  0.0 -1.0
2016-01-01 00:01:30+00:00 -1.0  0.0 -1.0
2016-01-01 00:02:30+00:00 -1.0  0.0 -1.0
...                        ...  ...  ...
2016-12-31 23:57:30+00:00 -2.0 -1.0 -2.0
2016-12-31 23:58:30+00:00 -2.0 -1.0 -2.0
2016-12-31 23:59:30+00:00 -2.0 -1.0 -2.0
[527040 rows x 3 columns]
[site=CAR/BSRN latitude=44.0830° longitude=5.0590° elevation=100.0 m]

>>> sdf.solpos.zenith  # solar zenith angle
time
2016-01-01 00:00:30+00:00    158.666033
2016-01-01 00:01:30+00:00    158.630072
2016-01-01 00:02:30+00:00    158.592202
                                ...    
2016-12-31 23:57:30+00:00    158.713107
2016-12-31 23:58:30+00:00    158.683687
2016-12-31 23:59:30+00:00    158.652329
Length: 527040, dtype: float64
[site=CAR/BSRN latitude=44.0830° longitude=5.0590° elevation=100.0 m]

>>> sdf.solpos.sunrise(units="utc")  # sunrise time, UTC
time
2016-01-01 00:00:30+00:00   2016-01-01 07:37:23.580818129
2016-01-01 00:01:30+00:00   2016-01-01 07:37:23.564837855
2016-01-01 00:02:30+00:00   2016-01-01 07:37:23.548856487
                                         ...             
2016-12-31 23:57:30+00:00   2017-01-01 07:37:05.570849828
2016-12-31 23:58:30+00:00   2017-01-01 07:37:05.553684227
2016-12-31 23:59:30+00:00   2017-01-01 07:37:05.536517540
Length: 527040, dtype: datetime64[ns]
[site=CAR/BSRN latitude=44.0830° longitude=5.0590° elevation=100.0 m]

>>> sdf.lta.ghi  # clear-sky ghi assuming a long-term average clear-sky atmosphere
time
2016-01-01 00:00:30+00:00    0.0
2016-01-01 00:01:30+00:00    0.0
2016-01-01 00:02:30+00:00    0.0
                            ... 
2016-12-31 23:57:30+00:00    0.0
2016-12-31 23:58:30+00:00    0.0
2016-12-31 23:59:30+00:00    0.0
Length: 527040, dtype: float64
[site=CAR/BSRN latitude=44.0830° longitude=5.0590° elevation=100.0 m]

>>> sdf.cda.ghi  # idem, but for a clean and dry clear-sky atmosphere
time
2016-01-01 00:00:30+00:00    0.0
2016-01-01 00:01:30+00:00    0.0
2016-01-01 00:02:30+00:00    0.0
                            ... 
2016-12-31 23:57:30+00:00    0.0
2016-12-31 23:58:30+00:00    0.0
2016-12-31 23:59:30+00:00    0.0
Length: 527040, dtype: float64
[site=CAR/BSRN latitude=44.0830° longitude=5.0590° elevation=100.0 m]

>>> sdf.clearsky.ghi  # idem, but using a preset clear-sky atmosphere from sparta-solar
time
2016-01-01 00:00:30+00:00    0.0
2016-01-01 00:01:30+00:00    0.0
2016-01-01 00:02:30+00:00    0.0
                            ... 
2016-12-31 23:57:30+00:00    0.0
2016-12-31 23:58:30+00:00    0.0
2016-12-31 23:59:30+00:00    0.0
Name: ghi, Length: 527040, dtype: float64
[site=CAR/BSRN latitude=44.0830° longitude=5.0590° elevation=100.0 m]

>>> sdf.clearsky.compute(  # ad-hoc non-cached calculation
...     atmosphere="crs_soda",
...     model="SPARTA")
                     dni  dhi  dif  ghi  csi
time                                        
2016-01-01 00:00:30  0.0  0.0  0.0  0.0  0.0
2016-01-01 00:01:30  0.0  0.0  0.0  0.0  0.0
2016-01-01 00:02:30  0.0  0.0  0.0  0.0  0.0
...                  ...  ...  ...  ...  ...
2016-12-31 23:57:30  0.0  0.0  0.0  0.0  0.0
2016-12-31 23:58:30  0.0  0.0  0.0  0.0  0.0
2016-12-31 23:59:30  0.0  0.0  0.0  0.0  0.0
[527040 rows x 5 columns]
[site=CAR/BSRN latitude=44.0830° longitude=5.0590° elevation=100.0 m]
```

- 𝘴𝘰𝘭𝘢𝘳𝘱𝘢𝘯𝘥𝘢𝘴 is **shipped with BSRN high-level data retrieval** and parsing utilities. When BSRN data is requested for the first time, it is downloaded, parsed, and archived locally in `parquet` format for fast subsequent access.

```python
>>> from solarpandas.origin import bsrn

>>> year_table = bsrn.data_availability(update="auto", as_year_table=True)
>>> print(year_table)
site |    9    0    0    1    1    2    2 
     |    5    0    5    0    5    0    5 
-----+------------------------------------
abs  |                              ######
aes  |                                    
ale  |             ###########            
asp  |    ##########################      
bar  | ###############################    
ber  | ######################  ##  #      
bil  |  ###########################       
...    ...

# 2) load station metadata (cached locally)
>>> meta = bsrn.load_metadata(update="auto")

# 3) load BSRN measurements for one station/year
>>> sdf = bsrn.load_data(
...     site="car",
...     years=2016,
...     logical_record="LR0100",
...     group="essential")
                           ghi  dni  dif
time                                    
2016-01-01 00:00:30+00:00 -1.0  0.0 -1.0
2016-01-01 00:01:30+00:00 -1.0  0.0 -1.0
2016-01-01 00:02:30+00:00 -1.0  0.0 -1.0
...                        ...  ...  ...
2016-12-31 23:57:30+00:00 -2.0 -1.0 -2.0
2016-12-31 23:58:30+00:00 -2.0 -1.0 -2.0
2016-12-31 23:59:30+00:00 -2.0 -1.0 -2.0
[527040 rows x 3 columns]
[site=CAR/BSRN latitude=44.0830° longitude=5.0590° elevation=100.0 m]
```

- It has **built-in quality-control workflows** enhanced with a tailored qc-specific ExtensionDType,  `qcflag`. The QC workflow is memory-cached and the 𝘴𝘰𝘭𝘢𝘳𝘱𝘢𝘯𝘥𝘢𝘴's `qcflag` dtype provides direct access to QC-specific methods via the `.flag` accessor.

```python
>>> sdf = sp.sample_data.load_carpentras_data()
>>> sdf.qc.tests  # perform the tests and return them
                           ghi_ppl  dif_ppl  ...  closure  trackeroff
time                                         ...                     
2016-01-01 00:00:30+00:00        0        0  ...        0           0
2016-01-01 00:01:30+00:00        0        0  ...        0           0
2016-01-01 00:02:30+00:00        0        0  ...        0           0
...                            ...      ...  ...      ...         ...
2016-12-31 23:57:30+00:00        0        0  ...        0           0
2016-12-31 23:58:30+00:00        0        0  ...        0           0
2016-12-31 23:59:30+00:00        0        0  ...        0           0
[527040 rows x 13 columns]
[site=CAR/BSRN latitude=44.0830° longitude=5.0590° elevation=0.0 m]

>>> sdf.qc.ghi_ppl  # access individual tests
time
2016-01-01 00:00:30+00:00    0
2016-01-01 00:01:30+00:00    0
2016-01-01 00:02:30+00:00    0
                            ..
2016-12-31 23:57:30+00:00    0
2016-12-31 23:58:30+00:00    0
2016-12-31 23:59:30+00:00    0
Name: ghi_ppl, Length: 527040, dtype: qcflag
[site=CAR/BSRN latitude=44.0830° longitude=5.0590° elevation=0.0 m]

>>> sdf.qc.ghi_ppl.dtype  # tests data have a special dtype `qcflag`
QCFlagDType()

# the type `qcflag` provides specific functionalities throught the `.flag` accessor
>>> sdf.qc.ghi_ppl.flag.counts()  # all data points in this dataset pass this test (by default, night time is excluded)
PASSED            265417
NOT_VERIFIABLE      1653
Name: count, dtype: int64

# and additional plotting methods:
>>> sdf.qc.ghi_ppl.flag.pieplot()
>>> sdf.qc.ghi_ppl.flag.heatmap()
>>> sdf.qc.ghi_ppl.flag.plot(sdf)

# 4) bolean masks from sets of individual tests
>>> failed_ghi = sdf.qc.failed(component="ghi")
>>> passed_all = sdf.qc.passed()

# 5) mask failed data points
>>> sdf_masked = sdf.qc.mask_failed(component="ghi")
>>> sdf.qc.heatmap(component="ghi")
```

- It provides specialized plotting helpers for solar datasets through the ``.solarplot`` accessor.

```python
# diurnal line plot
>>> fig1 = sdf.solarplot.diurnal(column="ghi")

# date-time heatmap
>>> fig2 = sdf.solarplot.heatmap(column="ghi", time_ref="tst", twilight_line=True)
```

## Installation

With pip:

```bash
pip install solarpandas
```

and with [uv](https://docs.astral.sh/uv/):

```bash
uv add solarpandas
```

Find further details in the [documentation](https://jararias.github.io/solarpandas).