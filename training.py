import pandas as pd
import numpy as np
from utils.cleaning import col_drop, fit_cleaning_stats, apply_cleaning
from utils.ordinal_encoding import fit_ordinal_medians, apply_ordinal_encoding
from utils.feature_engineering import engeneering_feature, filter_outliers
from utils.onehot_encoding import fit_onehot_categories, apply_onehot
from utils.persistence import saving_models, save_preprocessing_artifacts
from utils.metrics import print_results
from sklearn.model_selection import train_test_split
from models.linear_regression import MyLinearRegression
from models.ridge import MyRidge
from models.gradient_boosting import MyGradientBoost
from models.random_forest import MyRandomForest
from models.xgboost import MyXGBoost
from models.decision_tree import MyDecisionTree


def main():
    #import dataset
    df = pd.read_csv('./dataset/properties_final_irene.csv')
    
    #Dropping columns
    df = col_drop(df)
    
    #splitting dataset before cleaning
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Filter outliers: compute thresholds on train, apply to both
    train_df, test_df = filter_outliers(train_df, test_df)
    
    #Compute all statistics for numerical columns ONLY on train_df
    stats = fit_cleaning_stats(train_df) 
    
    #Apply statistics computed on train to dfs
    train_df = apply_cleaning(train_df, stats)
    test_df = apply_cleaning(test_df, stats)
    
    #Compute all statistics for categorical columns ONLY on train_df
    ordinal_medians = fit_ordinal_medians(train_df)
    
    #Apply statistics computed on train to dfs
    train_df = apply_ordinal_encoding(train_df, ordinal_medians)
    test_df = apply_ordinal_encoding(test_df, ordinal_medians)

    #Features engeneeringr
    train_df = engeneering_feature(train_df)
    test_df = engeneering_feature(test_df)
    
    #Compute all statistics for ONLY on train_df
    onehot_categories = fit_onehot_categories(train_df)
    
    #Apply statistics computed on train to dfs
    train_df = apply_onehot(train_df, onehot_categories)
    test_df = apply_onehot(test_df, onehot_categories)

    X_train = train_df.drop(columns='price').values
    X_test = test_df.drop(columns='price').values
    y_train = np.log1p(train_df["price"])
    y_test = np.log1p(test_df["price"])
    
    print(X_train.shape)
    
    #Models
    models = {
    "LinReg": MyLinearRegression(X_train, X_test, y_train, y_test),
    "Ridge": MyRidge(X_train, X_test, y_train, y_test),
    "GradBoost": MyGradientBoost(X_train, X_test, y_train, y_test),
    "XGBoost": MyXGBoost(X_train, X_test, y_train, y_test),
    "RandForest": MyRandomForest(X_train, X_test, y_train, y_test),
    "DecisionTree": MyDecisionTree(X_train, X_test, y_train, y_test),
    }
    
    #Saving preprocessing artifacts (needed for predict.py)
    feature_columns = list(train_df.drop(columns='price').columns)
    save_preprocessing_artifacts(stats, ordinal_medians, onehot_categories, feature_columns)


    #Savinig into models_trained
    saving_models(models)
    
    #Results
    print_results(models)
    
    
if __name__ == "__main__":
    main()