# Installation

## Requirements

- Python 3.13 or newer.
- An internet connection is required the first time BSRN data or station
  metadata is fetched from the remote FTP server. All subsequent access is
  served from a local Parquet cache.

## Install from PyPI

```bash
pip install solarpandas
```

## Install with uv

```bash
uv pip install solarpandas
```

## Install from Source

```bash
git clone https://github.com/jararias/solarpandas.git
cd solarpandas
pip install -e .
```

## Development Setup

Clone the repository, install in editable mode and sync the development
dependency group:

```bash
git clone https://github.com/jararias/solarpandas.git
cd solarpandas
pip install -e .
uv sync --group dev
```

## Verify Installation

Run the following snippet to confirm the package loads and the bundled
sample data is accessible:

```python
import solarpandas as sp

sdf = sp.sample_data.load_carpentras_data()
print(sdf.shape)      # (525600, 3) — one year at 1-minute resolution
print(sdf.latitude)   # 44.083
print(sdf.columns)    # Index(['ghi', 'dni', 'dif'], ...)
```

## Local Database Path

BSRN data downloaded from the FTP server is stored in a platform-specific
user data directory. To inspect or change the default path:

```python
from solarpandas.origin.bsrn import get_database_path

print(get_database_path())
# e.g. /home/user/.local/share/solarpandas/bsrn
```

The path can be overridden via the `bsrn.data_dir` configuration option:

```python
import solarpandas as sp

sp.set_option("bsrn.data_dir", "/data/bsrn")
```
