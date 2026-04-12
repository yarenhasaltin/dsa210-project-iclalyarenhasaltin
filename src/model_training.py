"""
Model training for productivity_score prediction.
Trains and compares Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, and XGBoost.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

from .utils import TARGET_COLUMN, OUTPUT_DIR, MODEL_ARTIFACT_PATH, PREPROCESSOR_ARTIFACT_PATH, ensure_output_dir

# Optional XGBoost
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def _get_models():
    """Return dict of name -> model (with scaling for linear models)."""
    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ]),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0)),
        ]),
        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Lasso(alpha=0.1)),
        ]),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
    return models


def train_and_compare(
    X,
    y,
    feature_names=None,
    test_size=0.2,
    random_state=42,
    scale_for_tree_models=False,
):
    """
    Train multiple regression models and compare by RMSE, MAE, R².
    Returns: results DataFrame, best model name, best model object, fitted scaler (if any), X_train, X_test, y_train, y_test.
    Tree models typically don't need scaling; linear models are wrapped in Pipeline with StandardScaler.
    """
    if feature_names is not None:
        X = X[feature_names].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale features for linear models; tree models get raw X
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    results = []
    fitted_models = {}

    for name, model in _get_models().items():
        use_scaled = isinstance(model, Pipeline)
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test
        try:
            model.fit(X_tr, y_train)
            pred = model.predict(X_te)
            if isinstance(model, Pipeline):
                reg = model.named_steps["reg"]
            else:
                reg = model
            fitted_models[name] = {
                "model": model,
                "used_scaled": use_scaled,
                "predict": lambda mod=model, xs=X_te: mod.predict(xs),
            }
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})
        except Exception as e:
            warnings.warn(f"Model {name} failed: {e}")
            results.append({"Model": name, "RMSE": np.nan, "MAE": np.nan, "R2": np.nan})

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    # Pick best model that actually fitted (in case some failed)
    best_name = None
    for _, row in results_df.iterrows():
        if row["Model"] in fitted_models and not pd.isna(row["R2"]):
            best_name = row["Model"]
            break
    if best_name is None:
        raise RuntimeError("No model fitted successfully. Check data and feature names.")
    best_info = fitted_models[best_name]
    best_model = best_info["model"]

    # Store scaler and feature order for use in recommendation engine and evaluation
    preprocessor = {
        "scaler": scaler,
        "feature_names": X_train.columns.tolist(),
        "best_uses_scaled": fitted_models[best_name]["used_scaled"],
    }

    return {
        "results_df": results_df,
        "best_model_name": best_name,
        "best_model": best_model,
        "preprocessor": preprocessor,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
    }


def get_feature_importance_tree(model, feature_names):
    """
    Extract feature importance from tree-based model (Random Forest, GB, XGBoost).
    Returns series of name -> importance or None if not available.
    """
    reg = model
    if isinstance(model, Pipeline):
        reg = model.named_steps.get("reg", model)
    if hasattr(reg, "feature_importances_"):
        return pd.Series(reg.feature_importances_, index=feature_names).sort_values(ascending=False)
    return None


def get_coefficients_linear(model, feature_names):
    """Get coefficients for linear models (Linear, Ridge, Lasso)."""
    if isinstance(model, Pipeline):
        reg = model.named_steps["reg"]
    else:
        reg = model
    if hasattr(reg, "coef_"):
        return pd.Series(reg.coef_, index=feature_names)
    return None


def save_best_model(best_model, preprocessor):
    """Save best model and preprocessor to disk."""
    import joblib
    ensure_output_dir()
    joblib.dump(best_model, MODEL_ARTIFACT_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_ARTIFACT_PATH)
