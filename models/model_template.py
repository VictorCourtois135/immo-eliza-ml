import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,
    median_absolute_error, explained_variance_score, max_error
)


class ModelTemplate:
    """Classe de base : centralise toutes les métriques communes aux modèles."""

    def _predict_train_real(self):
        y_pred_train = self.model.predict(self.X_train)
        return np.expm1(y_pred_train)

    @property
    def y_train_real(self):
        return np.expm1(self.y_train)

    # --- Scores R² ---
    def model_score(self):
        return self.model.score(self.X_test, self.y_test)

    def model_score_train(self):
        return self.model.score(self.X_train, self.y_train)

    def adjusted_r2(self):
        r2 = self.model_score()
        n = len(self.y_test)
        p = self.X_train.shape[1]
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    # --- Overfitting ---
    def overfit_gap(self):
        return self.model_score_train() - self.model_score()

    def overfit_ratio(self):
        train = self.model_score_train()
        test = self.model_score()
        return (train - test) / train if train != 0 else np.nan

    # --- Prédictions ---
    def model_predict(self):
        y_pred = self.model.predict(self.X_test)
        return np.expm1(y_pred)

    # --- Cross-validation ---
    def cross_val_scores(self):
        cv_stable = KFold(n_splits=10, shuffle=True, random_state=42)
        return cross_val_score(self.model, self.X_train, self.y_train, cv=cv_stable, scoring="r2", n_jobs=-1)

    def cross_val_mean(self):
        return self.cross_val_scores().mean()

    def cross_val_std(self):
        return self.cross_val_scores().std()

    # --- Métriques d'erreur ---
    def mean_absolute(self):
        return mean_absolute_error(self.y_test_real, self.model_predict())

    def median_absolute(self):
        return median_absolute_error(self.y_test_real, self.model_predict())

    def mean_squarred(self):
        return mean_squared_error(self.y_test_real, self.model_predict())

    def mean_squarred_root(self):
        return np.sqrt(mean_squared_error(self.y_test_real, self.model_predict()))

    def mean_absolute_percentage(self):
        return mean_absolute_percentage_error(self.y_test_real, self.model_predict())

    def explained_variance(self):
        return explained_variance_score(self.y_test_real, self.model_predict())

    def max_error_metric(self):
        return max_error(self.y_test_real, self.model_predict())