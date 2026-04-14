#!/usr/bin/env python3
"""
Smart Student Productivity Advisor: Measuring Digital Distraction and
Recommending Productivity Improvements with Machine Learning.

Full pipeline: load data, EDA, preprocessing, feature engineering,
modeling, digital distraction analysis, recommendation engine.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# make imports work from project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

from src.utils import (
    DATA_PATH,
    TARGET_COLUMN,
    OUTPUT_DIR,
    DIGITAL_DISTRACTION_FEATURES,
    POSITIVE_LIFESTYLE_FEATURES,
    ensure_output_dir,
    get_data_path,
)
from src.data_preprocessing import (
    load_data,
    basic_checks,
    handle_missing_values,
    encode_gender,
    remove_outliers_iqr,
    prepare_for_model,
)
from src.feature_engineering import engineer_all_features, get_feature_columns_for_model
from src.model_training import (
    train_and_compare,
    get_feature_importance_tree,
    get_coefficients_linear,
    save_best_model,
)
from src.evaluation import (
    model_comparison_bar_chart,
    actual_vs_predicted_plot,
    residual_plot,
    residual_distribution_plot,
    qq_plot_residuals,
    feature_importance_chart,
    permutation_importance_plot,
    recommendation_improvement_chart,
    print_metrics_summary,
)
from src.recommendation_engine import recommend_improvements, format_recommendations_output


# =============================================================================
# PART 2 — DATA LOADING AND BASIC CHECKS
# =============================================================================
def run_data_loading():
    # quick load + sanity checks
    print("\n" + "=" * 60)
    print("PART 2 — DATA LOADING AND BASIC CHECKS")
    print("=" * 60)
    df, is_synthetic = load_data(project_root=SCRIPT_DIR)
    print("Dataset loaded." + (" (Synthetic data)" if is_synthetic else ""))
    basic_checks(df)
    df = handle_missing_values(df, strategy="drop")
    return df


# =============================================================================
# PART 3 — EXPLORATORY DATA ANALYSIS
# =============================================================================
def run_eda(df):
    print("\n" + "=" * 60)
    print("PART 3 — EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    ensure_output_dir()

    # missing profile (helps for future real datasets)
    missing_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    missing_pct.plot(kind="bar", ax=ax, color="slategray")
    ax.set_title("Missing values by column (%)")
    ax.set_ylabel("Percent")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "missing_profile.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Missing profile plot generated.")

    # quick univariate plots
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    key_vars = [c for c in numeric_cols if c != "student_id" and c in df.columns][:12]
    if key_vars:
        fig, axes = plt.subplots(3, 4, figsize=(14, 10))
        axes = axes.flatten()
        for i, col in enumerate(key_vars):
            if i < len(axes):
                axes[i].hist(df[col].dropna(), bins=25, edgecolor="black", alpha=0.7)
                axes[i].set_title(col, fontsize=9)
        for j in range(len(key_vars), len(axes)):
            axes[j].set_visible(False)
        plt.suptitle("Univariate: Distributions of key variables", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "distributions.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("Distribution plots generated.")

    # boxplots for some key vars
    box_vars = [TARGET_COLUMN, "study_hours_per_day", "sleep_hours", "phone_usage_hours", "stress_level", "focus_score"]
    box_vars = [c for c in box_vars if c in df.columns]
    if box_vars:
        fig, ax = plt.subplots(figsize=(10, 5))
        df[box_vars].boxplot(ax=ax)
        plt.xticks(rotation=45, ha="right")
        plt.title("Boxplots: outlier detection")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "boxplots.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("Boxplots generated.")

    if "gender" in df.columns:
        print("\nGender value counts:\n", df["gender"].value_counts())
        # simple split by gender
        if TARGET_COLUMN in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df, x="gender", y=TARGET_COLUMN, ax=ax)
            ax.set_title("Productivity score by gender")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "productivity_by_gender.png"), dpi=150, bbox_inches="tight")
            plt.close()
            print("Gender comparison plot generated.")

    # small pairplot sample for quick structure check
    pair_cols = [
        TARGET_COLUMN, "study_hours_per_day", "sleep_hours",
        "phone_usage_hours", "social_media_hours", "focus_score"
    ]
    pair_cols = [c for c in pair_cols if c in df.columns]
    if len(pair_cols) >= 3:
        sample_n = min(250, len(df))
        pair_df = df[pair_cols].sample(sample_n, random_state=42)
        g = sns.pairplot(pair_df, corner=True, plot_kws={"alpha": 0.5, "s": 18})
        g.fig.suptitle("Pairplot sample (key vars)", y=1.02)
        g.fig.savefig(os.path.join(OUTPUT_DIR, "pairplot_key_vars.png"), dpi=150, bbox_inches="tight")
        plt.close(g.fig)
        print("Pairplot generated.")

    # correlation heatmap
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap="RdBu_r", center=0, square=True, ax=ax)
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Correlation heatmap generated.")

    # scatter vs target
    scatter_vars = [
        "study_hours_per_day", "sleep_hours", "phone_usage_hours", "social_media_hours",
        "youtube_hours", "gaming_hours", "stress_level", "focus_score",
        "attendance_percentage", "final_grade",
    ]
    scatter_vars = [c for c in scatter_vars if c in df.columns and c != TARGET_COLUMN]
    n_plots = len(scatter_vars)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = np.atleast_2d(axes)
    for i, col in enumerate(scatter_vars):
        r, c = i // n_cols, i % n_cols
        axes[r, c].scatter(df[col], df[TARGET_COLUMN], alpha=0.5)
        axes[r, c].set_xlabel(col)
        axes[r, c].set_ylabel(TARGET_COLUMN)
        axes[r, c].set_title(f"{TARGET_COLUMN} vs {col}")
    for i in range(len(scatter_vars), n_rows * n_cols):
        r, c = i // n_cols, i % n_cols
        axes[r, c].set_visible(False)
    plt.suptitle("Productivity score vs key variables", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "scatter_productivity.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Scatter plots generated.")

    # distraction vars vs target
    dist_cols = [c for c in DIGITAL_DISTRACTION_FEATURES if c in df.columns]
    if dist_cols:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
        for i, col in enumerate(dist_cols[:4]):
            axes[i].scatter(df[col], df[TARGET_COLUMN], alpha=0.6)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel(TARGET_COLUMN)
            axes[i].set_title(f"Digital distraction: {col}")
        plt.suptitle("Digital distraction vs productivity_score", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "digital_distraction_vs_productivity.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("Distraction scatter plots generated.")

    # quick correlation summary
    if TARGET_COLUMN in df.columns:
        corr_with_target = df.select_dtypes(include=[np.number]).corr()[TARGET_COLUMN].drop(TARGET_COLUMN, errors="ignore")
        corr_with_target = corr_with_target.sort_values(ascending=False)
        print("\n--- EDA Summary: Correlation with productivity_score ---")
        print("Strongest positive:", corr_with_target.head(5).to_string())
        print("Strongest negative:", corr_with_target.tail(5).to_string())

        # easier to read than full matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        corr_with_target.sort_values().tail(15).plot(kind="barh", ax=ax, color="teal")
        ax.set_title("Top positive correlations with productivity_score")
        ax.set_xlabel("Correlation")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "top_correlations_with_productivity.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("Correlation bar chart generated.")

        # formal tests if scipy exist
        try:
            from scipy.stats import pearsonr, spearmanr
            print("\n--- Statistical tests (Pearson/Spearman) ---")
            test_cols = [
                "phone_usage_hours", "social_media_hours", "youtube_hours", "gaming_hours",
                "study_hours_per_day", "sleep_hours", "exercise_minutes", "focus_score"
            ]
            for col in test_cols:
                if col in df.columns:
                    x = df[col].values
                    y = df[TARGET_COLUMN].values
                    pr, pp = pearsonr(x, y)
                    sr, sp = spearmanr(x, y)
                    print(
                        f"{col:24s} | Pearson r={pr:+.3f}, p={pp:.4g} | "
                        f"Spearman rho={sr:+.3f}, p={sp:.4g}"
                    )
        except Exception:
            print("\n(Scipy not available; skipped formal correlation tests.)")

    # rough outlier count report by IQR
    print("\n--- Outlier preview by IQR ---")
    for col in [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET_COLUMN]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        n_out = int(((df[col] < low) | (df[col] > high)).sum())
        if n_out > 0:
            print(f"{col:24s}: {n_out}")

    # distraction bucket analysis (practical for interventions)
    needed = set(DIGITAL_DISTRACTION_FEATURES + [TARGET_COLUMN])
    if needed.issubset(df.columns):
        temp = df.copy()
        temp["digital_distraction_score"] = temp[DIGITAL_DISTRACTION_FEATURES].sum(axis=1)
        temp["distraction_bucket"] = pd.qcut(temp["digital_distraction_score"], q=4, labels=["Low", "Med-Low", "Med-High", "High"])
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=temp, x="distraction_bucket", y=TARGET_COLUMN, ax=ax)
        ax.set_title("Productivity by digital distraction quartile")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "productivity_by_distraction_bucket.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("Distraction bucket plot generated.")
    return df


# =============================================================================
# PART 4 — PREPROCESSING AND FEATURE ENGINEERING
# =============================================================================
def run_preprocessing_and_engineering(df):
    print("\n" + "=" * 60)
    print("PART 4 — PREPROCESSING AND FEATURE ENGINEERING")
    print("=" * 60)
    df = encode_gender(df, copy=True)
    df = engineer_all_features(df, copy=True, include_leaky=True)

    # iqr outlier trim
    exclude_cols = ["student_id", TARGET_COLUMN] if "student_id" in df.columns else [TARGET_COLUMN]
    numeric_for_iqr = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
    n_before = len(df)
    df = remove_outliers_iqr(df, columns=numeric_for_iqr, factor=1.5)
    print(f"Outlier removal (IQR): {n_before} -> {len(df)} rows")

    # keep leakage-ish cols out
    model_cols = [c for c in get_feature_columns_for_model(include_final_grade=False, include_grade_gap=False) if c in df.columns]
    # just cols that are present
    X = df[model_cols].copy()
    y = df[TARGET_COLUMN]
    print("Features for model (no leakage):", model_cols)
    return df, X, y, model_cols


# =============================================================================
# PART 5 — MODELING
# =============================================================================
def run_modeling(X, y, feature_names):
    print("\n" + "=" * 60)
    print("PART 5 — MODELING: PRODUCTIVITY PREDICTION")
    print("=" * 60)
    out = train_and_compare(X, y, feature_names=feature_names, test_size=0.2, random_state=42)
    results_df = out["results_df"]
    print("\nModel comparison (sorted by R²):\n", results_df.to_string(index=False))
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_table.csv"), index=False)
    best_name = out["best_model_name"]
    best_model = out["best_model"]
    preprocessor = out["preprocessor"]
    X_test = out["X_test"]
    y_test = out["y_test"]
    X_test_scaled = out["X_test_scaled"]
    used_scaled = preprocessor["best_uses_scaled"]
    y_pred = best_model.predict(X_test_scaled if used_scaled else X_test)
    pd.DataFrame({"actual": y_test.values, "predicted": y_pred}).to_csv(
        os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False
    )
    print_metrics_summary(y_test, y_pred, best_name)
    save_best_model(best_model, preprocessor)
    return out, y_pred, y_test


# =============================================================================
# PART 5 continued — EVALUATION PLOTS
# =============================================================================
def run_evaluation_plots(out, y_pred, y_test):
    ensure_output_dir()
    model_comparison_bar_chart(out["results_df"])
    actual_vs_predicted_plot(y_test.values, y_pred, title="Actual vs Predicted (Best Model)")
    residual_plot(y_test.values, y_pred)
    residual_distribution_plot(y_test.values, y_pred)
    qq_plot_residuals(y_test.values, y_pred)
    best_model = out["best_model"]
    fn = out["preprocessor"]["feature_names"]
    imp = get_feature_importance_tree(best_model, fn)
    if imp is not None:
        feature_importance_chart(imp, title="Feature importance (best model)")
    # can be bit slow
    X_te = out["X_test_scaled"] if out["preprocessor"]["best_uses_scaled"] else out["X_test"]
    try:
        permutation_importance_plot(best_model, X_te, y_test, fn, n_repeats=5)
    except Exception as e:
        print("Permutation importance skipped:", e)
    print("Evaluation plots generated.")


# =============================================================================
# PART 6 — DIGITAL DISTRACTION IMPACT ANALYSIS
# =============================================================================
def run_digital_distraction_analysis(df, out):
    print("\n" + "=" * 60)
    print("PART 6 — DIGITAL DISTRACTION IMPACT ANALYSIS")
    print("=" * 60)
    # 1. Correlation of distraction features with productivity
    dist_cols = [c for c in DIGITAL_DISTRACTION_FEATURES + ["digital_distraction_score"] if c in df.columns]
    if dist_cols and TARGET_COLUMN in df.columns:
        corr_dist = df[dist_cols + [TARGET_COLUMN]].corr()[TARGET_COLUMN].drop(TARGET_COLUMN)
        print("Correlation with productivity_score (distraction):\n", corr_dist.sort_values().to_string())

    # 2. Feature importance from best model
    best_model = out["best_model"]
    fn = out["preprocessor"]["feature_names"]
    imp = get_feature_importance_tree(best_model, fn)
    if imp is not None:
        dist_imp = imp[imp.index.isin(DIGITAL_DISTRACTION_FEATURES + ["digital_distraction_score"])]
        if not dist_imp.empty:
            print("\nFeature importance (distraction-related):\n", dist_imp.to_string())

    # 3. Reduced model: distraction-only features
    reduced_features = [
        "phone_usage_hours", "social_media_hours", "youtube_hours", "gaming_hours",
        "breaks_per_day", "coffee_intake_mg", "stress_level",
    ]
    reduced_features = [f for f in reduced_features if f in fn]
    if len(reduced_features) >= 3:
        from sklearn.metrics import r2_score
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        X_full = out["X_train"]
        y_train = out["y_train"]
        X_te = out["X_test"]
        y_test = out["y_test"]
        X_red_train = X_full[reduced_features]
        X_red_test = X_te[reduced_features]
        model_red = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model_red.fit(X_red_train, y_train)
        pred_red = model_red.predict(X_red_test)
        r2_red = r2_score(y_test, pred_red)
        r2_full = out["results_df"].iloc[0]["R2"]
        print(f"\nReduced model (distraction-focused) R²: {r2_red:.4f}")
        print(f"Full model R²: {r2_full:.4f}")
        print("Distraction-only features explain a meaningful but smaller portion of productivity.")

    # 4. Interpretation
    print("\n--- Interpretation ---")
    print("Which distraction variable hurts productivity most: check correlation and feature importance above.")
    print("Positive habits that offset distraction: study_hours, sleep_hours, exercise_minutes, focus_score, attendance.")

    # 5. Optional SHAP (if available)
    try:
        import shap
        X_te = out["X_test_scaled"] if out["preprocessor"]["best_uses_scaled"] else out["X_test"]
        explainer = shap.Explainer(out["best_model"], X_te[: min(100, len(X_te))])
        shap_vals = explainer.shap_values(X_te[: min(100, len(X_te))])
        if hasattr(shap_vals, "shape") and len(shap_vals.shape) >= 2:
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            shap_imp = pd.Series(mean_abs_shap, index=out["preprocessor"]["feature_names"]).sort_values(ascending=False)
            print("\nSHAP mean |impact| (top 10):\n", shap_imp.head(10).to_string())
    except Exception as e:
        print("\n(SHAP skipped:", str(e)[:60] + ")")


# =============================================================================
# PART 7 & 8 — RECOMMENDATION ENGINE AND SAMPLE CASES
# =============================================================================
def run_recommendations(df, out, feature_names):
    print("\n" + "=" * 60)
    print("PART 7 & 8 — RECOMMENDATION ENGINE AND SAMPLE CASES")
    print("=" * 60)
    best_model = out["best_model"]
    preprocessor = out["preprocessor"]
    preprocessor["best_uses_scaled"] = out["preprocessor"]["best_uses_scaled"]

    # sample rows from data
    df_with_features = df.copy()
    for idx in [0, len(df) // 4, len(df) // 2]:
        if idx >= len(df_with_features):
            continue
        row = df_with_features.iloc[idx]
        profile = row.to_dict()
        try:
            current_pred, recs = recommend_improvements(profile, best_model, preprocessor, feature_names, top_k=5)
            print(f"\n--- Student sample (row index {idx}) ---")
            print(f"Profile (sample): study_hours={profile.get('study_hours_per_day'):.1f}, sleep={profile.get('sleep_hours'):.1f}, phone={profile.get('phone_usage_hours'):.1f}")
            print(format_recommendations_output(current_pred, recs))
            if recs:
                recommendation_improvement_chart(current_pred, recs, save_path=os.path.join(OUTPUT_DIR, f"recommendation_sample_{idx}.png"))
        except Exception as e:
            print(f"Recommendation failed for row {idx}:", e)

    # one handmade example
    manual_profile = {
        "student_id": "manual_1",
        "age": 21,
        "gender": "Female",
        "study_hours_per_day": 3,
        "sleep_hours": 6,
        "phone_usage_hours": 5,
        "social_media_hours": 3,
        "youtube_hours": 2,
        "gaming_hours": 1,
        "breaks_per_day": 4,
        "coffee_intake_mg": 200,
        "exercise_minutes": 15,
        "assignments_completed": 8,
        "attendance_percentage": 75,
        "stress_level": 7,
        "focus_score": 4,
        "final_grade": 55,
        "productivity_score": 50,
    }
    # make sure encoded/engineered fields exist
    from src.data_preprocessing import encode_gender
    from src.feature_engineering import update_engineered_in_profile
    df_man = pd.DataFrame([manual_profile])
    df_man = encode_gender(df_man, copy=False)
    df_man = df_man.drop(columns=["gender"], errors="ignore")
    manual_profile = df_man.iloc[0].to_dict()
    update_engineered_in_profile(manual_profile)
    try:
        current_pred, recs = recommend_improvements(manual_profile, best_model, preprocessor, feature_names, top_k=5)
        print("\n--- Manually created example student ---")
        print("Profile: study=3h, sleep=6h, phone=5h, social_media=3h, exercise=15min, stress=7, focus=4")
        print(format_recommendations_output(current_pred, recs))
        if recs:
            recommendation_improvement_chart(current_pred, recs, save_path=os.path.join(OUTPUT_DIR, "recommendation_manual.png"))
    except Exception as e:
        print("Recommendation failed for manual profile:", e)


# =============================================================================
# PART 10 — REPORT SUMMARY
# =============================================================================
def print_report_summary():
    print("\n" + "=" * 60)
    print("PART 10 — REPORT SUMMARY (for course report)")
    print("=" * 60)
    summary = """
Problem definition: Predict student productivity_score from lifestyle and academic behavior;
analyze impact of digital distraction; recommend lifestyle changes to improve productivity.

Data preprocessing: Schema validation, missing/duplicate handling, gender encoding, IQR-based outlier removal.

Feature engineering: digital_distraction_score, healthy_lifestyle_score, academic_engagement_score,
study_to_phone_ratio, stress_focus_balance, break_efficiency_score, caffeine_stress_interaction,
grade_productivity_gap (excluded from model to avoid leakage).

Modeling: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost compared;
best model selected by R²/RMSE/MAE; standardized features for linear models.

Digital distraction analysis: Correlation and feature importance show which distractions hurt most;
reduced model (distraction-only) vs full model comparison.

Recommendation engine: Scenario-based suggestions (reduce phone/social/gaming/youtube, increase sleep/exercise/study)
with realistic bounds; outputs ranked recommendations and expected productivity gain.

Evaluation: Metrics table, actual vs predicted plot, residual plot, feature importance, permutation importance.

Future improvements: More data, temporal features, classification for productivity tiers, A/B testing of recommendations.
"""
    print(summary)


# =============================================================================
# MAIN
# =============================================================================
def main():
    ensure_output_dir()
    df = run_data_loading()
    df = run_eda(df)
    df, X, y, feature_names = run_preprocessing_and_engineering(df)
    out, y_pred, y_test = run_modeling(X, y, feature_names)
    run_evaluation_plots(out, y_pred, y_test)
    run_digital_distraction_analysis(df, out)
    run_recommendations(df, out, feature_names)
    print_report_summary()
    print("\nDone.")


if __name__ == "__main__":
    main()
