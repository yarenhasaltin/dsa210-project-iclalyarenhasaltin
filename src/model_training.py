"""
Advanced model training for productivity_score prediction.

The pipeline compares regularized linear models, robust regression, tree ensembles,
gradient boosting, and an optional stacking ensemble / XGBoost variant.
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import (
    ElasticNetCV,
    HuberRegressor,
    LassoCV,
    LinearRegression,
    RidgeCV,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .utils import (
    MODEL_ARTIFACT_PATH,
    OUTPUT_DIR,
    PREPROCESSOR_ARTIFACT_PATH,
    ensure_output_dir,
)

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def _safe_mape(y_true, y_pred, eps=1e-6):
    """MAPE with protection against zeros in the target."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _regression_metrics(y_true, y_pred, n_features):
    """Compute the metric bundle used across candidate models."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = _safe_mape(y_true, y_pred)
    n = len(y_true)
    if n > n_features + 1:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    else:
        adj_r2 = np.nan
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Adj_R2": float(adj_r2) if not np.isnan(adj_r2) else np.nan,
        "MAPE": mape,
    }


def _scaled_pipeline(regressor):
    """Wrap a regressor in a standard-scaling pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", regressor),
        ]
    )


def _get_models(random_state=42, include_xgboost=False, cv_splits=3):
    """Return advanced candidate models."""
    alphas = np.logspace(-4, 4, 25)
    sparse_alphas = np.logspace(-4, 1, 30)
    models = {
        "Linear Regression": _scaled_pipeline(LinearRegression()),
        "Ridge CV": _scaled_pipeline(RidgeCV(alphas=alphas, cv=cv_splits)),
        "Lasso CV": _scaled_pipeline(
            LassoCV(alphas=sparse_alphas, cv=cv_splits, max_iter=20000, random_state=random_state)
        ),
        "ElasticNet CV": _scaled_pipeline(
            ElasticNetCV(
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
                alphas=sparse_alphas,
                cv=cv_splits,
                max_iter=20000,
                random_state=random_state,
            )
        ),
        "Huber Regression": _scaled_pipeline(HuberRegressor(max_iter=500)),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=8,
            max_iter=250,
            l2_regularization=0.05,
            random_state=random_state,
        ),
    }
    if include_xgboost and HAS_XGB:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
        )
    return models


def _fit_single_model(name, model, X_train, X_test, y_train, y_test, X_cv, y_cv, cv):
    """Fit one candidate model and gather test/CV metrics."""
    started = time.time()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = _regression_metrics(y_test, pred, n_features=X_train.shape[1])

    try:
        cv_rmse = -cross_val_score(
            clone(model),
            X_cv,
            y_cv,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=None,
        )
        cv_mae = -cross_val_score(
            clone(model),
            X_cv,
            y_cv,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=None,
        )
        cv_r2 = cross_val_score(
            clone(model),
            X_cv,
            y_cv,
            cv=cv,
            scoring="r2",
            n_jobs=None,
        )
        metrics.update(
            {
                "CV_RMSE_mean": float(np.mean(cv_rmse)),
                "CV_RMSE_std": float(np.std(cv_rmse)),
                "CV_MAE_mean": float(np.mean(cv_mae)),
                "CV_MAE_std": float(np.std(cv_mae)),
                "CV_R2_mean": float(np.mean(cv_r2)),
                "CV_R2_std": float(np.std(cv_r2)),
            }
        )
    except Exception:
        metrics.update(
            {
                "CV_RMSE_mean": np.nan,
                "CV_RMSE_std": np.nan,
                "CV_MAE_mean": np.nan,
                "CV_MAE_std": np.nan,
                "CV_R2_mean": np.nan,
                "CV_R2_std": np.nan,
            }
        )

    metrics.update(
        {
            "Model": name,
            "FitSeconds": round(time.time() - started, 3),
            "model": model,
            "pred": pred,
        }
    )
    return metrics


def _build_stacking_model(fitted_rows, model_catalog, random_state=42):
    """Build a stacking regressor from the strongest base candidates."""
    eligible = []
    for row in fitted_rows:
        model_name = row["Model"]
        if model_name == "Stacking Ensemble":
            continue
        score = row.get("CV_R2_mean")
        if pd.isna(score):
            score = row.get("R2", -np.inf)
        eligible.append((score, model_name))
    eligible.sort(reverse=True)
    top_names = []
    for _, name in eligible:
        if name not in top_names:
            top_names.append(name)
        if len(top_names) == 3:
            break
    if len(top_names) < 2:
        return None

    estimators = []
    for idx, model_name in enumerate(top_names):
        estimators.append((f"model_{idx+1}", clone(model_catalog[model_name])))

    return StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(alphas=np.logspace(-4, 4, 25), cv=3),
        passthrough=True,
        n_jobs=-1,
    )


def train_and_compare(
    X,
    y,
    feature_names=None,
    test_size=0.2,
    random_state=42,
    include_xgboost=False,
    cv_splits=3,
    cv_max_rows=5000,
    include_stacking=True,
):
    """Train advanced regressors and compare them with test and CV metrics."""
    if feature_names is not None:
        X = X[feature_names].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    if cv_max_rows is not None and len(X_train) > cv_max_rows:
        cv_idx = X_train.sample(n=cv_max_rows, random_state=random_state).index
        X_cv = X_train.loc[cv_idx]
        y_cv = y_train.loc[cv_idx]
    else:
        X_cv = X_train
        y_cv = y_train

    model_catalog = _get_models(
        random_state=random_state,
        include_xgboost=include_xgboost,
        cv_splits=cv_splits,
    )

    fitted_rows = []
    fitted_models = {}
    predictions = {}

    for name, model in model_catalog.items():
        try:
            row = _fit_single_model(
                name=name,
                model=model,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                X_cv=X_cv,
                y_cv=y_cv,
                cv=cv,
            )
            predictions[name] = row.pop("pred")
            fitted_models[name] = row.pop("model")
            fitted_rows.append(row)
        except Exception as e:
            warnings.warn(f"Model {name} failed: {e}")
            fitted_rows.append(
                {
                    "Model": name,
                    "RMSE": np.nan,
                    "MAE": np.nan,
                    "R2": np.nan,
                    "Adj_R2": np.nan,
                    "MAPE": np.nan,
                    "CV_RMSE_mean": np.nan,
                    "CV_RMSE_std": np.nan,
                    "CV_MAE_mean": np.nan,
                    "CV_MAE_std": np.nan,
                    "CV_R2_mean": np.nan,
                    "CV_R2_std": np.nan,
                    "FitSeconds": np.nan,
                }
            )

    if include_stacking:
        stacking_model = _build_stacking_model(fitted_rows, model_catalog, random_state=random_state)
        if stacking_model is not None:
            try:
                row = _fit_single_model(
                    name="Stacking Ensemble",
                    model=stacking_model,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    X_cv=X_cv,
                    y_cv=y_cv,
                    cv=cv,
                )
                predictions["Stacking Ensemble"] = row.pop("pred")
                fitted_models["Stacking Ensemble"] = row.pop("model")
                fitted_rows.append(row)
            except Exception as e:
                warnings.warn(f"Stacking Ensemble failed: {e}")

    results_df = pd.DataFrame(fitted_rows)
    results_df["SelectionScore"] = results_df["CV_R2_mean"].fillna(results_df["R2"])
    results_df = results_df.sort_values(
        ["SelectionScore", "R2", "RMSE"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    best_name = None
    for _, row in results_df.iterrows():
        if row["Model"] in fitted_models and not pd.isna(row["R2"]):
            best_name = row["Model"]
            break
    if best_name is None:
        raise RuntimeError("No model fitted successfully. Check data and feature names.")

    best_model = fitted_models[best_name]
    best_pred = predictions[best_name]
    preprocessor = {
        "scaler": None,
        "feature_names": X_train.columns.tolist(),
        "best_uses_scaled": False,
        "best_model_name": best_name,
    }

    return {
        "results_df": results_df,
        "best_model_name": best_name,
        "best_model": best_model,
        "best_predictions": best_pred,
        "preprocessor": preprocessor,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train.copy(),
        "X_test_scaled": X_test.copy(),
    }


def get_feature_importance_tree(model, feature_names):
    """
    Extract feature importance from tree-based or wrapped estimators.
    Returns series of name -> importance or None if not available.
    """
    reg = model
    if isinstance(model, Pipeline):
        reg = model.named_steps.get("reg", model)
    if hasattr(reg, "feature_importances_"):
        return pd.Series(reg.feature_importances_, index=feature_names).sort_values(ascending=False)
    return None


def get_coefficients_linear(model, feature_names):
    """Get coefficients for linear-style models when available."""
    reg = model
    if isinstance(model, Pipeline):
        reg = model.named_steps.get("reg", model)
    if hasattr(reg, "coef_"):
        coef = np.ravel(reg.coef_)
        if len(coef) == len(feature_names):
            return pd.Series(coef, index=feature_names).sort_values(
                key=lambda s: np.abs(s),
                ascending=False,
            )
    return None


def save_best_model(best_model, preprocessor):
    """Save best model and preprocessor to disk."""
    import joblib

    ensure_output_dir()
    joblib.dump(best_model, MODEL_ARTIFACT_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_ARTIFACT_PATH)
