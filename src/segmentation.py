import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN

def _resolve_col(df: pd.DataFrame, preferred: str, aliases: list[str]) -> str:
    """Return the first matching column among preferred + aliases (case-insensitive)."""
    if preferred in df.columns:
        return preferred
    lower_map = {c.lower(): c for c in df.columns}
    for cand in [preferred] + aliases:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise KeyError(f"Could not find any of these columns: { [preferred] + aliases } in CSV headers: {list(df.columns)}")

def rfm_table(df: pd.DataFrame,
              date_col: str,
              customer_col: str,
              revenue_col: str,
              invoice_col: str) -> pd.DataFrame:
    """
    Builds RFM table with robust column resolution.
    """
    customer_col = _resolve_col(df, customer_col, ["Customer ID", "Customer_Id", "Customer Id"])
    date_col     = _resolve_col(df, date_col, ["Invoice Date", "InvoiceDate"])
    revenue_col  = _resolve_col(df, revenue_col, ["Revenue", "Amount", "Total"])
    invoice_col  = _resolve_col(df, invoice_col, ["InvoiceNo", "Invoice No", "Invoice"])

    max_date = df[date_col].max()
    rfm = (df.groupby(customer_col)
             .agg(
                 Recency=(date_col, lambda s: (max_date - s.max()).days),
                 Frequency=(invoice_col, "nunique"),
                 Monetary=(revenue_col, "sum")
             )
          ).reset_index()
    return rfm

def kmeans_clusters(rfm: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    rfm["KMeansCluster"] = km.fit_predict(X)
    return rfm
    

def dbscan_clusters(rfm: pd.DataFrame, eps: float = 0.5, min_samples: int = 15) -> pd.DataFrame:
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    rfm["DBSCANCluster"] = db.fit_predict(X)
    return rfm
