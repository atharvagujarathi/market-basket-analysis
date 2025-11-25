import argparse, yaml
from pathlib import Path
import pandas as pd
from data_prep import load_data, clean_data, basket_dataframe, daily_sales
from mba import run_mba
from segmentation import rfm_table, kmeans_clusters, dbscan_clusters
from forecasting import sarima_forecast, xgb_forecast
from utils import save_csv, plot_series, plot_pred_vs_actual
import warnings
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    csv_path = cfg["data"]["csv_path"]
    date_col = cfg["data"]["date_column"]
    qty_col = cfg["data"]["quantity_column"]
    price_col = cfg["data"]["price_column"]
    invoice_col = cfg["data"]["invoice_column"]
    customer_col = cfg["data"]["customer_id_column"]
    stock_col = cfg["data"]["stockcode_column"]

    artifacts = Path(cfg["outputs"]["artifacts_dir"])
    reports = Path(cfg["outputs"]["reports_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    print("Loading data")
    df = load_data(csv_path)
    print("Cleaning data")
    df = clean_data(df, date_col, qty_col, price_col, invoice_col, customer_col)

    print("Running Market Basket Analysis")
    basket = basket_dataframe(df, invoice_col=invoice_col, stock_col=stock_col)
    rules = run_mba(
        basket_df=basket,
        min_support=cfg["mba"]["min_support"],
        min_confidence=cfg["mba"]["min_confidence"],
        use_fpgrowth=cfg["mba"]["use_fpgrowth"]
    )
    top_rules = rules.head(cfg["mba"]["top_rules"])
    save_csv(top_rules, artifacts / "mba_top_rules.csv")
    print(f"Saved MBA rules -> {artifacts / 'mba_top_rules.csv'}")

        # ----- CLUSTERING ACCURACY METRICS -----
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    print("Calculating clustering accuracy metrics...")

    # Prepare scaled data again
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # K-Means Silhouette Score
    km_silhouette = silhouette_score(X, rfm_km["KMeansCluster"])
    print("K-Means Silhouette Score:", km_silhouette)

    # DBSCAN Silhouette Score (ignore noise = -1)
    mask = rfm_db["DBSCANCluster"] != -1
    if mask.sum() > 1:
        db_silhouette = silhouette_score(X[mask], rfm_db.loc[mask, "DBSCANCluster"])
    else:
        db_silhouette = None
        print("DBSCAN Silhouette Score: Not applicable (too much noise)")

    print("Clustering metrics calculated.")


    print("Building RFM table")
    rfm = rfm_table(
    df,
    date_col=date_col,
    customer_col=customer_col,
    revenue_col="Revenue",
    invoice_col=invoice_col
)

    rfm_km = kmeans_clusters(rfm.copy(), k=cfg["segmentation"]["kmeans_k"])
    rfm_db = dbscan_clusters(rfm.copy(), eps=cfg["segmentation"]["dbscan_eps"], min_samples=cfg["segmentation"]["dbscan_min_samples"])
    save_csv(rfm_km, artifacts / "rfm_kmeans.csv")
    save_csv(rfm_db, artifacts / "rfm_dbscan.csv")
    print(f"Saved RFM clusters -> {artifacts / 'rfm_kmeans.csv'}, {artifacts / 'rfm_dbscan.csv'}")

    # This code is for Forecasting
    print("Preparing daily sales...")
    daily = daily_sales(df, date_col=date_col)
    plot_series(daily, "Daily Revenue", reports / "daily_revenue.png")

    test_days = int(cfg["forecasting"]["test_days"])
    sarima_order = tuple(cfg["forecasting"]["sarima_order"])
    sarima_seasonal = tuple(cfg["forecasting"]["sarima_seasonal_order"])

    train = daily.iloc[:-test_days].copy()
    test  = daily.iloc[-test_days:].copy()

    print("Training SARIMA")
    sar_preds, sar_rmse = sarima_forecast(train, test, sarima_order, sarima_seasonal)
    plot_pred_vs_actual(test["ds"], test["y"], sar_preds, f"SARIMA Forecast (RMSE={sar_rmse:.2f})",
                        reports / "sarima_forecast.png")
    
    sar_mae = mean_absolute_error(test["y"], sar_preds)
    sar_r2 = r2_score(test["y"], sar_preds)

    print(f"SARIMA -> RMSE={sar_rmse:.2f}, MAE={sar_mae:.2f}, R2={sar_r2:.3f}")


    print("Training XGBoost")
    xgb_params = cfg["forecasting"]["xgb"]
    from forecasting import add_calendar_features, make_lags
    # Recreate features to get aligned y_true for test
    from forecasting import add_calendar_features, make_lags
    df_feat = add_calendar_features(daily)
    df_feat = make_lags(df_feat, cfg["forecasting"]["lags"])
    y_true = df_feat["y"].iloc[-test_days:].values
    ds_true = df_feat["ds"].iloc[-test_days:].values

    xgb_preds, xgb_rmse = xgb_forecast(daily, test_days, cfg["forecasting"]["lags"], xgb_params)
    plot_pred_vs_actual(ds_true, y_true, xgb_preds, f"XGBoost Forecast (RMSE={xgb_rmse:.2f})",
                        reports / "xgb_forecast.png")
    
    # XGBoost additional metrics
    xgb_mae = mean_absolute_error(y_true, xgb_preds)
    xgb_r2 = r2_score(y_true, xgb_preds)

    print(f"XGBoost -> RMSE={xgb_rmse:.2f}, MAE={xgb_mae:.2f}, R2={xgb_r2:.3f}")


    metrics = pd.DataFrame([
    {"model": "SARIMA",  "rmse": sar_rmse, "mae": sar_mae, "r2": sar_r2},
    {"model": "XGBoost", "rmse": xgb_rmse, "mae": xgb_mae, "r2": xgb_r2}
    ])

    save_csv(metrics, artifacts / "forecast_metrics.csv")
    print("Done. See artifacts/ and reports/ for outputs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
