
"""Sample datasets bundled with solarpandas for demos and tests."""

from pathlib import Path

from loguru import logger

from ..iohelpers import read_parquet  # noqa: F401

logger.disable(__name__)
logger = logger.opt(colors=True)


def load_carpentras_data():
    """Load bundled Carpentras BSRN sample data.

    Returns
    -------
    SolarDataFrame
        Pre-packaged sample dataset stored in Parquet format.

    Examples
    --------
    >>> import solarpandas as sp
    >>> sdf = sp.sample_data.load_carpentras_data()

    Notes
    -----
    This helper is intended for demos, examples and quick local checks.
    """
    this_dir = Path(__file__).absolute().parent
    filename = this_dir / "car_bsrn_2016.parquet"
    data = read_parquet(filename)
    logger.success(f"Carpentras BSRN data loaded from {filename.relative_to(this_dir.parent)}")
    return data
