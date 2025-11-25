import pandas as pd
import numpy as np
from typing import List, Tuple
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dow"] = out["ds"].dt.dayofweek
    out["dom"] = out["ds"].dt.day
    out["month"] = out["ds"].dt.month
    out["is_weekend"] = out["dow"].isin([5,6]).astype(int)
    return out

def make_lags(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    out = df.copy()
    for L in lags:
        out[f"lag_{L}"] = out["y"].shift(L)
    out = out.dropna().reset_index(drop=True)
    return out

def sarima_forecast(train_df: pd.DataFrame,
                    test_df: pd.DataFrame,
                    order: Tuple[int,int,int],
                    seasonal_order: Tuple[int,int,int,int]) -> Tuple[np.ndarray, float]:
    mod = SARIMAX(train_df["y"],
                  order=order,
                  seasonal_order=seasonal_order,
                  enforce_stationarity=False,
                  enforce_invertibility=False)
    fit = mod.fit(disp=False)
    preds = fit.forecast(steps=len(test_df))
    rmse = sqrt(mean_squared_error(test_df["y"], preds))
    return preds.values, rmse

def xgb_forecast(daily_df: pd.DataFrame,
                 test_days: int,
                 lags: List[int],
                 params: dict) -> Tuple[np.ndarray, float]:
    df_feat = add_calendar_features(daily_df)
    df_feat = make_lags(df_feat, lags)

    train = df_feat.iloc[:-test_days].copy()
    test  = df_feat.iloc[-test_days:].copy()

    X_train = train.drop(columns=["y", "ds"])
    y_train = train["y"]
    X_test  = test.drop(columns=["y", "ds"])
    y_test  = test["y"]

    model = XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, preds))
    return preds, rmse
