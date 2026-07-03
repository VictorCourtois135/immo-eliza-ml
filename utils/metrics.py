"""
Model evaluation utilities: prints a comparison table of key
regression metrics (R², overfitting indicators, error metrics) across
several trained models.
"""


def print_results(models: dict) -> None:
    """
    Print a formatted comparison table of regression metrics for each
    model in `models` (dict of name -> fitted model instance exposing
    the BaseRegressorModel interface).
    """
    header = f"{'Metric':<18}" + "".join(f"{name:>15}" for name in models)
    sep = "-" * len(header)

    def oob_row_value(m):
        """OOB score only exists for bagging models (e.g. Random Forest)."""
        if hasattr(m, "oob_score"):
            return f"{m.oob_score():>15.6f}"
        return f"{'N/A':>15}"

    rows = [
        header, sep,
        f"{'R² (test)':<18}"      + "".join(f"{m.model_score():>15.6f}" for m in models.values()),
        f"{'R² (train)':<18}"     + "".join(f"{m.model_score_train():>15.6f}" for m in models.values()),
        f"{'R² ajusté':<18}"      + "".join(f"{m.adjusted_r2():>15.6f}" for m in models.values()),
        f"{'OOB Score':<18}"      + "".join(oob_row_value(m) for m in models.values()),
        f"{'CV R² moyen':<18}"    + "".join(f"{m.cross_val_mean():>15.6f}" for m in models.values()),
        f"{'CV R² std':<18}"      + "".join(f"{m.cross_val_std():>15.6f}" for m in models.values()),
        f"{'Overfit (gap)':<18}"  + "".join(f"{m.overfit_gap():>15.6f}" for m in models.values()),
        f"{'Overfit (%)':<18}"    + "".join(f"{m.overfit_ratio()*100:>14.2f}%" for m in models.values()),
        f"{'Explained Var.':<18}" + "".join(f"{m.explained_variance():>15.6f}" for m in models.values()),
        sep,
        f"{'MAE':<18}"       + "".join(f"{m.mean_absolute():,.0f} €".rjust(15) for m in models.values()),
        f"{'Median AE':<18}" + "".join(f"{m.median_absolute():,.0f} €".rjust(15) for m in models.values()),
        f"{'MAPE':<18}"      + "".join(f"{m.mean_absolute_percentage()*100:,.2f} %".rjust(15) for m in models.values()),
        f"{'MSE':<18}"       + "".join(f"{m.mean_squarred():,.0f} €".rjust(15) for m in models.values()),
        f"{'RMSE':<18}"      + "".join(f"{m.mean_squarred_root():,.0f} €".rjust(15) for m in models.values()),
        f"{'Max Error':<18}" + "".join(f"{m.max_error_metric():,.0f} €".rjust(15) for m in models.values()),
    ]

    print("\n".join(rows))