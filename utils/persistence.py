"""
Model persistence utilities.
"""
import joblib


def saving_models(models: dict) -> None:
    """
    Serialize each trained model to disk under ./models_trained/<name>.pkl
    """
    for name, m in models.items():
        joblib.dump(m, f"./models_trained/{name}.pkl")
        
def save_preprocessing_artifacts(stats: dict, ordinal_medians: dict,
                                  onehot_categories: dict, feature_columns: list) -> None:
    """
    Save all fitted preprocessing statistics (computed on train only),
    plus the exact training feature column order, so predict.py can
    apply the same transformations and produce a correctly-ordered
    feature vector for the model.
    """
    joblib.dump(stats, "./models_trained/cleaning_stats.pkl")
    joblib.dump(ordinal_medians, "./models_trained/ordinal_medians.pkl")
    joblib.dump(onehot_categories, "./models_trained/onehot_categories.pkl")
    joblib.dump(feature_columns, "./models_trained/feature_columns.pkl")