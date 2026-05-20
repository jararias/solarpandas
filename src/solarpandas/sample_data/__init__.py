
from pathlib import Path

from loguru import logger

from ..base import read_parquet  # noqa: F401

logger.disable(__name__)
logger = logger.opt(colors=True)


def load_carpentras_data():
    this_dir = Path(__file__).absolute().parent
    filename = this_dir / "car_bsrn_2016.parquet"
    data = read_parquet(filename)
    logger.success(f"Carpentras BSRN data loaded from {filename.relative_to(this_dir.parent)}")
    return data
