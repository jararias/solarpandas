# Quick References

## Minimal Import

```python
import solarpandas as sp
```

## Load Sample Data

```python
sdf = sp.sample_data.load_carpentras_data()
```

## Read and Write

```python
sdf.to_csv("data.csv")
sdf.to_parquet("data.parquet")

sdf_csv = sp.read_csv("data.csv")
sdf_parquet = sp.read_parquet("data.parquet")
```

## Solar Position

```python
sdf.solpos.zenith
sdf.solpos.azimuth
sdf.solpos.cosz
```

## Clear-Sky

```python
sdf.clearsky.ghi
sdf.clearsky.dni
sdf.clearsky.dif
```

## Quality Control

```python
qc = sdf.qc
qc.tests
qc.failed(component="ghi")
qc.passed(component="ghi")
qc.mask_failed(component="ghi")
```

## BSRN Core Functions

```python
from solarpandas.origin.bsrn import data_availability, load_metadata, load_data
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
