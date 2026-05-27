
from importlib.resources import files

import matplotlib.pyplot as plt

def register():
    path = files("solarpandas.mplstyles")
    styles = plt.style.core.read_style_directory(path)
    for key, value in styles.items():
        plt.style.library[f"solarpandas-{key}"] = value

