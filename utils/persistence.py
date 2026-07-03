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