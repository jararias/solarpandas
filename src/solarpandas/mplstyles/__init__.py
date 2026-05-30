
"""Registration helpers and constants for bundled Matplotlib styles."""

from importlib.resources import files

import matplotlib.pyplot as plt

def register():
    """Register bundled Matplotlib styles under the ``solarpandas-*`` prefix.

    Notes
    -----
    Calling this function multiple times is safe; style keys are overwritten
    with the same values.

    Examples
    --------
    >>> from solarpandas.mplstyles import register
    >>> register()
    """
    path = files("solarpandas.mplstyles")
    styles = plt.style.core.read_style_directory(path)
    for key, value in styles.items():
        plt.style.library[f"solarpandas-{key}"] = value

QC_COLOR_FAILED = "#d46c4c"  # light red
QC_COLOR_PASSED = "#e6f2ff"  # light blue
QC_COLOR_NOT_VERIFIABLE = "#ffffcc"  # light yellow
