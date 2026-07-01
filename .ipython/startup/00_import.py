# ruff: noqa: F401, E402

from IPython import get_ipython
from IPython.core.magic import register_line_magic

ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

import numpy as np
import pylab as pl
import pandas as pd

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

pl.ion()

try:
    import solarpandas as sp
except Exception as exc:
    red_cross = "\033[91m\u2718\033[0m"
    print(f"`solarpandas` could not be imported {red_cross}")
    raise exc

green_tick = "\033[92m\u2714\033[0m"  # red cross
print(f"`solarpandas` imported with __version__ = {sp.__version__} {green_tick}")
del(green_tick)

data = sp.sample_data.load_carpentras_data()
