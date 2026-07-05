import os

project = "fade"

folders = [
    f"{project}/core",
    f"{project}/pipeline",
    f"{project}/utils",
]

files = {
f"{project}/__init__.py": "",
f"{project}/core/__init__.py": "",
f"{project}/pipeline/__init__.py": "",
f"{project}/utils/__init__.py": "",

f"{project}/core/atoms.py": """
import pandas as pd
import numpy as np

def compute_atoms(df):
    df = df.copy()
    df["return_1h"] = df["close"].pct_change()
    df["return_6h"] = df["close"].pct_change(6)
    df["volatility"] = df["return_1h"].rolling(24).std()
    df["volume_z"] = (df["volume"] - df["volume"].rolling(24).mean()) / df["volume"].rolling(24).std()
    df["trend"] = df["close"].rolling(12).mean() - df["close"].rolling(24).mean()
    return df
""",

f"{project}/pipeline/main.py": """
import pandas as pd
from fade.core.atoms import compute_atoms

df = pd.read_csv("btc.csv")

df = compute_atoms(df)

print(df.tail())
print('FADE RUN COMPLETE')
"""
}

for f in folders:
    os.makedirs(f, exist_ok=True)

for path, content in files.items():
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

print("FADE PROJECT CREATED")