import pandas as pd
import numpy as np
from typing import Tuple

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="latin1")
    return df

def clean_data(df: pd.DataFrame,
               date_col: str,
               qty_col: str,
               price_col: str,
               invoice_col: str,
               customer_col: str) -> pd.DataFrame:

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce").fillna(0)

    df = df[(df[qty_col] > 0) & (df[price_col] > 0)]

    if customer_col in df.columns:
        df = df[~df[customer_col].isna()]
        try:
            df[customer_col] = df[customer_col].astype("int64")
        except Exception:
            df[customer_col] = df[customer_col].astype("int64", errors="ignore")

    df["Revenue"] = df[qty_col] * df[price_col]

    return df

def basket_dataframe(df: pd.DataFrame,
                     invoice_col: str,
                     stock_col: str) -> pd.DataFrame:
    """
    Creates a basket dataframe: rows=invoices, columns=items.
    """
    basket = (df.groupby([invoice_col, stock_col])["Revenue"]
                .sum()
                .unstack(fill_value=0))
    basket = (basket > 0).astype(int)
    return basket

def daily_sales(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    daily = df.set_index(date_col).resample("D")["Revenue"].sum().reset_index()
    daily.columns = ["ds", "y"]
    return daily