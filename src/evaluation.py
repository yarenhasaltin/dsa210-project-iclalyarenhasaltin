"""
Evaluation and visualization for the Smart Student Productivity Advisor.
Metrics, actual vs predicted, residuals, feature importance, permutation importance.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

from .utils import OUTPUT_DIR, ensure_output_dir
from .model_training import get_feature_importance_tree, get_coefficients_linear


def model_comparison_bar_chart(results_df, save_path=None):
    """Bar chart comparing R² (and optionally RMSE) across models."""
    ensure_output_dir()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(results_df))
    width = 0.35
    ax.bar(x - width / 2, results_df["R2"], width, label="R²", color="steelblue")
    ax.bar(x + width / 2, results_df["RMSE"] / results_df["RMSE"].max(), width, label="RMSE (norm)", color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Model"], rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: R² and Normalized RMSE")
    ax.legend()
    plt.tight_layout()
    if save_path is None:
        save_path = f"{OUTPUT_DIR}/model_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def actual_vs_predicted_plot(y_true, y_pred, title="Actual vs Predicted (Best Model)", save_path=None):
    """Scatter plot of actual vs predicted productivity score."""
    ensure_output_dir()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.5)
    lims = [min(y_true.min(), y_pred.min()) - 2, max(y_true.max(), y_pred.max()) + 2]
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set_xlabel("Actual productivity_score")
    ax.set_ylabel("Predicted productivity_score")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    plt.tight_layout()
    if save_path is None:
        save_path = f"{OUTPUT_DIR}/actual_vs_predicted.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def residual_plot(y_true, y_pred, save_path=None):
    """Residuals vs predicted values."""
    ensure_output_dir()
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted productivity_score")
    ax.set_ylabel("Residual")
    ax.set_title("Residual Plot (Best Model)")
    plt.tight_layout()
    if save_path is None:
        save_path = f"{OUTPUT_DIR}/residuals.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def feature_importance_chart(importance_series, title="Feature Importance", save_path=None, top_n=20):
    """Horizontal bar chart for feature importance."""
    if importance_series is None or importance_series.empty:
        return None
    ensure_output_dir()
    s = importance_series.head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(6, len(s) * 0.3)))
    ax.barh(range(len(s)), s.values, color="steelblue", alpha=0.8)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    if save_path is None:
        save_path = f"{OUTPUT_DIR}/feature_importance.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def permutation_importance_plot(model, X, y, feature_names, n_repeats=10, random_state=42, save_path=None):
    """Compute and plot permutation importance."""
    ensure_output_dir()
    X_arr = X if hasattr(X, "values") else np.array(X)
    if hasattr(model, "predict"):
        result = permutation_importance(model, X_arr, y, n_repeats=n_repeats, random_state=random_state)
        imp = pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=True)
        imp = imp.tail(20)
        fig, ax = plt.subplots(figsize=(8, max(5, len(imp) * 0.25)))
        ax.barh(range(len(imp)), imp.values, color="seagreen", alpha=0.8)
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels(imp.index, fontsize=9)
        ax.set_xlabel("Permutation importance (decrease in R²)")
        ax.set_title("Permutation Importance (Best Model)")
        plt.tight_layout()
        if save_path is None:
            save_path = f"{OUTPUT_DIR}/permutation_importance.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return save_path, imp
    return None, None


def recommendation_improvement_chart(current_score, recommendations, save_path=None):
    """
    recommendations: list of dicts with keys e.g. 'description', 'new_score', 'improvement'
    """
    if not recommendations:
        return None
    ensure_output_dir()
    labels = [r.get("description", f"Rec {i+1}")[:40] for i, r in enumerate(recommendations[:8])]
    improvements = [r.get("improvement", 0) for r in recommendations[:8]]
    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, improvements, color="teal", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Expected productivity improvement")
    ax.set_title("Top recommendations: expected gain in productivity score")
    plt.tight_layout()
    if save_path is None:
        save_path = f"{OUTPUT_DIR}/recommendation_improvement.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def print_metrics_summary(y_true, y_pred, model_name="Best model"):
    """Print RMSE, MAE, R² for a set of predictions."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name} — RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    return {"RMSE": rmse, "MAE": mae, "R2": r2}
