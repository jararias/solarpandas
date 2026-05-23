from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

# (X,I2,X,I4,2(3X,I4,X,F5.1,X,I4,X,I4),/
# 8X,2(3X,I4,X,F5.1,X,I4,X,I4),4X,F5.1,X,F5.1,X,I4)

# Tu string con los datos consecutivos del mes
texto_mes = (
    Path("c0100test.data").read_text().splitlines()
)  # Reemplaza con tu fuente de datos real

COLUMNS_LINE_1 = (
    "day",
    "minute",
    "global_horizontal",
    "global_horizontal_std",
    "global_horizontal_min",
    "global_horizontal_max",
    "direct_normal",
    "direct_normal_std",
    "direct_normal_min",
    "direct_normal_max",
)

COLUMNS_LINE_2 = (
    "diffuse_horizontal",
    "diffuse_horizontal_std",
    "diffuse_horizontal_min",
    "diffuse_horizontal_max",
    "downward_longwave",
    "downward_longwave_std",
    "downward_longwave_min",
    "downward_longwave_max",
    "air_temperature",
    "relative_humidity",
    "atmospheric_pressure",
)

# line_1: (X,I2,X,I4,2(3X,I4,X,F5.1,X,I4,X,I4),/
# 0         1         2         3         4         5
# 012345678901234567890123456789012345678901234567890123456789
#  I2 __I4   __I4 _F5.1 __I4 __I4   __I4 _F5.1 __I4 __I4
colspecs = [(1, 3), (4, 8), (11, 15), (16, 21), (22, 26), (27, 31), (34, 38), (39, 44), (45, 49), (50, 54)]
buffer = StringIO("\n".join(texto_mes[::2]))
line_1 = pd.read_fwf(buffer, colspecs=colspecs, header=None).set_axis(COLUMNS_LINE_1, axis=1)

# line_2: 8X,2(3X,I4,X,F5.1,X,I4,X,I4),4X,F5.1,X,F5.1,X,I4)
# 0         1         2         3         4         5         6         7
# 01234567890123456789012345678901234567890123456789012345678901234567890123456789
#            __I4 _F5.1 __I4 __I4   __I4 _F5.1 __I4 __I4    _F5.1 _F5.1 __I4
colspecs = [(11, 15), (16, 21), (22, 26), (27, 31), (34, 38), (39, 44), (45, 49), (50, 54), (58, 63), (64, 69), (70, 74)]
buffer = StringIO("\n".join(texto_mes[1::2]))
line_2 = pd.read_fwf(buffer, colspecs=colspecs, header=None).set_axis(COLUMNS_LINE_2, axis=1)

data = pd.concat([line_1, line_2], axis=1)

def check_day_consistency(df):
    # TODO: meter warnings
    day_number = df.name
    df = df.sort_values("minute")
    if (df.iloc[0]["minute"] == 1) and (df.iloc[-1]["minute"] == 1440) and (len(df) == 1440):
        df["minute"] = df["minute"] - 1  # Ajustamos para que el minuto 1 corresponda a 00:00
    df = df.loc[(df["minute"] >=0) & (df["minute"] < 1440)]
    hour, minute = np.divmod(df["minute"], 60)
    df["hour"] = hour
    df["minute"] = minute
    df = df.get(["hour", "minute"] + df.columns.drop(["hour", "minute"]).tolist())
    df = df.loc[(df["hour"] >= 0) & (df["hour"] < 24)]
    df = df.loc[(df["minute"] >= 0) & (df["minute"] < 60)]
    print(df)
    return df

x = data.groupby("day").apply(lambda df: check_day_consistency(df)).reset_index("day")
print(x)
