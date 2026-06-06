r"""Configuration loading and runtime option access for solarpandas.

This module initializes the user configuration file on first use, loads TOML
content into an in-memory dictionary, and exposes helpers to query or override
options during the current Python session.

Notes
-----
The persistent config file is stored in the platform-specific user config
directory and is named ``config.toml``.

Examples
--------
>>> from solarpandas.config import get_config_path, get_option, set_option
>>> config_path = get_config_path()
>>> email = get_option("crs_soda.user_email")
>>> set_option("solar-position.algorithm", "spa")
"""

# HOW DOES THIS CONFIG WORK? WHEN THE FLOW PASS THROUGH, IT CHECKS IF THE CONFIG
# FILE EXISTS PHYSICALLY IN LOCAL. IF NOT, IT CREATES IT WITH THE DEFAULT SET-UP.
# THEN, THE FILE IS READ AND STORED IN _GLOBAL_CONFIG.
# THE CONFIGURATION CAN BE CONSULTED WITH show_options AND get_option, AND EVEN
# MODIFIED WITH set_option. HOWEVER, CHANGES ONLY AFFECT THE CURRENT SESSION.
# TO PERSIST THE CHANGES THE USER MUST DO IT MANUALLY BY EDITING THE CONFIG FILE.
# THE PATH TO THE LOCAL FILE IS AVAILABLE IN get_config_path

from pathlib import Path
from typing import Any

import platformdirs
import tomlkit
from loguru import logger

logger.disable(__name__)
logger = logger.opt(colors=True)

_DEFAULT_CONFIG_TOML_ = """
[solar-position]  # table to set the sunwhere's options
algorithm = "psa"  # solar position algorithm
refraction = true
engine = "numexpr"

[clearsky]  # table to set the clearsky model and atmosphere
model = "SPARTA"  # clearsky model to use for irradiance estimation
atmosphere = "crs_soda"  # atmosphere dataset to use for clearsky calculations
lta_atmosphere = "merra2_lta"  # atmosphere dataset for long-term average clearsky
cda_atmosphere = "merra2_cda"  # atmosphere dataset for clear-day analysis

[bsrn]  # table for BSRN data retrieval settings
# data_dir = "."  # local directory to store BSRN data (leave empty for default)
server = "ftp.bsrn.awi.de"
"""


def get_config_path() -> Path:
    """Return the path of the user configuration file.

    Returns
    -------
    pathlib.Path
        Absolute path to ``config.toml`` in the platform-specific user
        configuration directory.
    """
    path = platformdirs.user_config_path(appname="solarpandas", ensure_exists=True)
    return path / "config.toml"


def _init_config_file():
    """Create the default config file from the built-in TOML template."""
    with get_config_path().open(mode="w") as f:
        f.write(_DEFAULT_CONFIG_TOML_)
    logger.success(
        f"user's config file initialized at <blue>{get_config_path()}</blue>"
    )


def _read_config_options() -> dict[str, Any]:
    """Read all options from the configuration file.

    If the configuration file does not exist, it initializes it with
    default placeholder values.

    Returns
    -------
    dict[str, Any]
        Parsed TOML content.
    """

    if not (config_path := get_config_path()).exists():
        _init_config_file()

    with config_path.open(mode="rb") as f:
        return tomlkit.load(f)


def reset_config_file():
    """Reset configuration to defaults by deleting and recreating the file.

    Notes
    -----
    This operation updates the in-memory global configuration immediately.
    """
    global _GLOBAL_CONFIG
    if get_config_path().exists():
        get_config_path().unlink()
        logger.success(f"config file {get_config_path()} deleted")
    _GLOBAL_CONFIG = _read_config_options()


_GLOBAL_CONFIG = _read_config_options()


def save_config(path: Path | None = None) -> None:
    """Persist the in-memory configuration to a TOML file.

    Parameters
    ----------
    path : pathlib.Path or None, default None
        Optional output path. When ``None``, the default location from
        :func:`get_config_path` is used.

    Notes
    -----
    Path instances in values are converted to POSIX strings before writing.
    """
    target = get_config_path() if path is None else Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    def _serialize(obj: Any):
        if isinstance(obj, Path):
            return obj.as_posix()
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(v) for v in obj]
        return obj

    serializable = _serialize(_GLOBAL_CONFIG)
    with target.open(mode="w", encoding="utf-8") as f:
        f.write(tomlkit.dumps(serializable))
    logger.success(f"config saved at <blue>{target}</blue>")


def load_config(path: Path | None = None, overwrite: bool = True) -> dict[str, Any]:
    """Load configuration from a TOML file and optionally overwrite global state.

    Parameters
    ----------
    path : pathlib.Path or None, default None
        Optional path to a TOML file. If ``None``, the default config path is used.
    overwrite : bool, default True
        If ``True``, replace ``_GLOBAL_CONFIG`` entirely. If ``False``, only
        missing top-level tables are inserted (shallow merge).

    Returns
    -------
    dict[str, Any]
        Loaded configuration dictionary.
    """
    cfg_path = get_config_path() if path is None else Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"config file not found: {cfg_path}")

    with cfg_path.open(mode="rb") as f:
        loaded = tomlkit.load(f)

    global _GLOBAL_CONFIG
    if overwrite:
        _GLOBAL_CONFIG = loaded
    else:
        for k, v in loaded.items():
            _GLOBAL_CONFIG.setdefault(k, v)

    logger.success(f"config loaded from <blue>{cfg_path}</blue>")
    return _GLOBAL_CONFIG


def show_config() -> None:
    """Print all current global options to the console.

    Notes
    -----
    Uses :func:`pprint.pprint` for a compact formatted output.
    """
    from pprint import pprint

    return pprint(_GLOBAL_CONFIG, indent=2, width=20)


def get_option(name: str, default: Any = None) -> Any:
    """Retrieve the value of a specific configuration option.

    Options are organized in tables (sections) within the TOML file.
    This function uses dot notation to access nested values.

    Parameters
    ----------
    name : str
        Option path in ``<table>.<option>`` format
        (for example, ``"solar-position.algorithm"``).
    default : Any, default None
        Value returned when the table or option is missing.

    Returns
    -------
    Any
        Option value, or ``default`` when missing. Options named ``data_dir``
        are returned as :class:`pathlib.Path`.

    Examples
    --------
    >>> from solarpandas.config import get_option
    >>> algorithm = get_option("solar-position.algorithm")
    >>> server = get_option("bsrn.server", default="ftp.bsrn.awi.de")
    >>> data_dir = get_option("bsrn.data_dir")  # returns a Path or None
    """
    table_name, option_name = name.split(".")
    if (table := _GLOBAL_CONFIG.get(table_name, None)) is None:
        logger.warning(f"missing table `{table_name}`")
        return default
    if (value := table.get(option_name, None)) is None:
        return default
    if option_name == "data_dir":
        return Path(value)
    return value


def set_option(name: str, value: Any) -> None:
    """Temporarily update a global option for the current session.

    Modifies configuration values in memory only. Changes are lost when
    the Python session ends. To make persistent changes, edit the
    config.toml file directly.

    Parameters
    ----------
    name : str
        Option path in ``<table>.<option>`` format.
    value : Any
        New value. For ``data_dir`` options, ``Path`` is converted to string.

    Examples
    --------
    >>> from solarpandas.config import set_option, get_option
    >>> set_option("solar-position.algorithm", "nrel")
    >>> get_option("solar-position.algorithm")
    'nrel'
    >>> from pathlib import Path
    >>> set_option("bsrn.data_dir", Path("/tmp/bsrn-cache"))

    Notes
    -----
    Changes are session-local. Call :func:`save_config` to persist them.
    """

    table_name, option_name = name.split(".")
    if _GLOBAL_CONFIG.get(table_name, None) is None:
        logger.warning(f"missing table `{table_name}`")
        return None
    if option_name == "data_dir" and isinstance(value, Path):
        value = value.as_posix()
    _GLOBAL_CONFIG[table_name][option_name] = value
