import numpy as np
import pandas as pd
import joblib

from utils.cleaning import apply_cleaning
from utils.ordinal_encoding import apply_ordinal_encoding
from utils.feature_engineering import engeneering_feature
from utils.onehot_encoding import apply_onehot

MODEL_NAME = "XGBoost"  # best-performing model based on training.py results


def load_artifacts():
    """Load the trained model and all preprocessing artifacts fitted on train."""
    model = joblib.load(f"./models_trained/{MODEL_NAME}.pkl")
    stats = joblib.load("./models_trained/cleaning_stats.pkl")
    ordinal_medians = joblib.load("./models_trained/ordinal_medians.pkl")
    onehot_categories = joblib.load("./models_trained/onehot_categories.pkl")
    feature_columns = joblib.load("./models_trained/feature_columns.pkl")
    return model, stats, ordinal_medians, onehot_categories, feature_columns


def build_dummy_input() -> pd.DataFrame:
    """
    Create a single new property as a DataFrame, mimicking the raw
    columns available right after col_drop() in training.py.
    Adjust these values to test different scenarios.
    """
    data = {
        "latitude": [50.8503],
        "longitude": [4.3517],
        "property_type": ["House"],
        "property_subtype": ["Villa"],
        "region": ["Brussels"],
        "province": ["Brussels"],
        "living_area_m2": [180.0],
        "bedrooms": [3],
        "bathrooms": [2],
        "facades": [4],
        "building_year": [1995],
        "garden_area_m2": [120.0],
        "has_garden": [1],
        "state_of_the_building": ["Normal"],
        "epc_score": ["C"],
    }
    return pd.DataFrame(data)


def preprocess_new_data(df, stats, ordinal_medians, onehot_categories, feature_columns):
    """
    Apply the exact same preprocessing steps used in training.py,
    reusing artifacts fitted on the training set only.
    """
    df = apply_cleaning(df, stats)
    df = apply_ordinal_encoding(df, ordinal_medians)
    df = engeneering_feature(df)
    df = apply_onehot(df, onehot_categories)
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


def predict_price(model_wrapper, X) -> float:
    """
    Predict the price (in €) for a single property.

    model_wrapper is an instance of MyXGBoost (or similar), which wraps
    the real fitted estimator in its `.model` attribute.
    """
    y_pred_log = model_wrapper.model.predict(X)
    y_pred_real = np.expm1(y_pred_log)
    return float(y_pred_real[0])


def main():
    model_wrapper, stats, ordinal_medians, onehot_categories, feature_columns = load_artifacts()

    new_house = build_dummy_input()
    processed = preprocess_new_data(new_house, stats, ordinal_medians, onehot_categories, feature_columns)

    X_new = processed.values
    predicted_price = predict_price(model_wrapper, X_new)

    print(f"Predicted price: {predicted_price:,.0f} €")


if __name__ == "__main__":
    main()