# Installation

## Requirements

- Python 3.13 or newer.

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

```bash
pip install -e .
uv sync --group dev
```

## Verify Installation

```python
import solarpandas as sp

sdf = sp.sample_data.load_carpentras_data()
print(sdf.shape)
```
