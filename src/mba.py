import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

def run_mba(basket_df: pd.DataFrame,
            min_support: float = 0.01,
            min_confidence: float = 0.3,
            use_fpgrowth: bool = True) -> pd.DataFrame:
    if use_fpgrowth:
        freq = fpgrowth(basket_df, min_support=min_support, use_colnames=True)
    else:
        freq = apriori(basket_df, min_support=min_support, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values(["lift", "confidence", "support"], ascending=False)
    return rules
