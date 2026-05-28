

import pandas as pd
from loguru import logger

logger.disable(__name__)
logger = logger.opt(colors=True)


# TODO: FUNCION PARA RELLENAR HUECOS

# TODO: FUNCION PARA COMPLETAR DIAS POR DELANTE Y POR DETRAS


def infer_time_step(df: pd.DataFrame) -> pd.Timedelta | None:
    """Infer time step from index of a DataFrame."""
    if (time_step := pd.infer_freq(df.index)) is None:
        logger.warning("Could not infer the time step from index using `pd.infer_freq`")
        time_step = df.index.diff().unique().drop(pd.NaT, errors="ignore")
        if len(time_step) == 0:
            logger.warning("Could not infer the time step from index taking the shortest "
                           "lag between consecutive rows")
            logger.error("No valid time steps found.")
            return None
    return time_step

