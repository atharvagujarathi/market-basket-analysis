# E-Commerce Data Mining Project

Market Basket Analysis, Customer Segmentation, and Sales Forecasting

## Quick Start

1. Run the bellow command to satisfy all the project requirements
   pip install -r requirements.txt

2. Put the Online Retail II CSV file at the path set in `config.yaml` (e.g., `data/online_retail_II.csv`).

3. python src/main.py --config config.yaml

Outputs (CSV/PNG/PKL) will be written to `artifacts/` and `reports/`.

## Notes

- Market Basket: Apriori & FP-Growth via `mlxtend`.
- Segmentation: RFM + KMeans/DBSCAN.
- Forecasting: SARIMA (statsmodels) + XGBoost with lag features.
- Plots are saved under `reports/`.
