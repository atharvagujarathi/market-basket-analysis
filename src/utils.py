import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def save_csv(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def plot_series(daily_df: pd.DataFrame, title: str, out_path: str):
    plt.figure()
    plt.plot(daily_df["ds"], daily_df["y"])
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_pred_vs_actual(ds, y_true, y_pred, title: str, out_path: str):
    plt.figure()
    plt.plot(ds, y_true, label="Actual")
    plt.plot(ds, y_pred, label="Predicted")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
