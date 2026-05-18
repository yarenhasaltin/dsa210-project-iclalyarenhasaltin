#!/usr/bin/env python3
"""
Comprehensive optimization and analysis pipeline.
Focuses on:
1. Leakage investigation
2. Feature selection and subset comparisons
3. Model interpretability with SHAP
4. Generation of actionable insights

Runs all analyses needed for the final comprehensive report.
"""

import os
import sys
import warnings

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import (
    DATA_PATH, TARGET_COLUMN, OUTPUT_DIR, ensure_output_dir,
    DIGITAL_DISTRACTION_FEATURES, POSITIVE_LIFESTYLE_FEATURES,
)
from src.data_preprocessing import (
    load_data, basic_checks, handle_missing_values,
    encode_gender, remove_outliers_iqr, prepare_for_model,
)
from src.feature_engineering import engineer_all_features, get_feature_columns_for_model
from src.model_training import train_and_compare, get_feature_importance_tree
from src.leakage_investigation import run_leakage_investigation
from src.feature_optimization import run_feature_subset_experiments

warnings.filterwarnings('ignore')


def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE ML OPTIMIZATION & ANALYSIS PIPELINE")
    print("="*80)
    
    ensure_output_dir()
    
    # =========================================================================
    # STEP 1: DATA LOADING AND PREPARATION
    # =========================================================================
    print("\n[STEP 1] Loading and preparing data...")
    df, is_synthetic = load_data(project_root=SCRIPT_DIR)
    print(f"Loaded {len(df)} records. Synthetic: {is_synthetic}")
    
    basic_checks(df)
    df = handle_missing_values(df, strategy="drop")
    df = encode_gender(df, copy=True)
    df = engineer_all_features(df, copy=True, include_leaky=True)
    
    # Remove outliers
    exclude_cols = ["student_id", TARGET_COLUMN]
    numeric_for_iqr = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
    df = remove_outliers_iqr(df, columns=numeric_for_iqr, factor=1.5)
    print(f"After preprocessing: {len(df)} records")
    
    # Get model features (without leakage)
    model_cols = [c for c in get_feature_columns_for_model(include_final_grade=False, include_grade_gap=False) if c in df.columns]
    X = df[model_cols].copy()
    y = df[TARGET_COLUMN]
    
    print(f"Features for analysis: {len(model_cols)}")
    print(f"Target mean: {y.mean():.4f}, std: {y.std():.4f}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # =========================================================================
    # STEP 2: LEAKAGE INVESTIGATION
    # =========================================================================
    print("\n[STEP 2] Running leakage investigation...")
    
    # First, train a baseline model to get feature importances for permutation importance
    print("  - Training baseline model for permutation importance...")
    out_baseline = train_and_compare(
        X, y,
        feature_names=model_cols,
        test_size=0.2,
        random_state=42,
        include_xgboost=False,
        cv_splits=3,
    )
    best_model = out_baseline["best_model"]
    X_test = out_baseline["X_test"]
    y_test = out_baseline["y_test"]
    
    print(f"  - Baseline best model R²: {out_baseline['results_df'].iloc[0]['R2']:.6f}")
    
    # Run leakage investigation with permutation importance from the trained model
    leakage_results = run_leakage_investigation(
        X, y, model_cols,
        model=best_model,
        X_test=X_test,
        y_test=y_test
    )
    
    consensus_features = leakage_results['consensus_series']
    
    # =========================================================================
    # STEP 3: FEATURE SUBSET EXPERIMENTS
    # =========================================================================
    print("\n[STEP 3] Running feature subset experiments...")
    subset_results, subset_predictions = run_feature_subset_experiments(
        df, X, y, model_cols, consensus_features
    )
    
    # Save subset results
    subset_results.to_csv(f"{OUTPUT_DIR}/feature_subset_optimization_results.csv", index=False)
    
    # =========================================================================
    # STEP 4: GENERATE SHAP EXPLANATIONS (if available)
    # =========================================================================
    print("\n[STEP 4] Generating SHAP explanations for top model...")
    try:
        import shap
        
        # Use the best GB model for SHAP (more stable than linear)
        X_train_top10 = subset_predictions['Top10'][0]
        X_test_top10 = subset_predictions['Top10'][1]
        top_10_features = X_train_top10.columns.tolist()
        
        print(f"  - Computing SHAP values for {len(top_10_features)} features...")
        
        # Train a final GB model for SHAP
        from sklearn.ensemble import GradientBoostingRegressor
        gb_final = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            min_samples_leaf=5, subsample=0.8, random_state=42
        )
        gb_final.fit(X_train_top10, y.iloc[X_train_top10.index])
        
        # SHAP values
        explainer = shap.TreeExplainer(gb_final)
        shap_values = explainer.shap_values(X_test_top10)
        
        # Save SHAP plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_top10, plot_type="bar", show=False)
        plt.title("SHAP Summary Plot - Feature Importance\n(Top 10 Features Model)")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/shap_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("  ✓ SHAP summary plot generated")
        
        # SHAP dependence plots for top 4 features
        top_features_for_shap = X_test_top10.columns[:4]
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for idx, feat in enumerate(top_features_for_shap):
            ax = axes.flatten()[idx]
            shap.dependence_plot(feat, shap_values, X_test_top10, ax=ax, show=False)
        plt.suptitle("SHAP Dependence Plots - Top 4 Features", fontsize=14, y=1.00)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/shap_dependence_plots.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("  ✓ SHAP dependence plots generated")
        
    except Exception as e:
        print(f"  ⚠ SHAP generation skipped: {e}")
    
    # =========================================================================
    # STEP 5: ACTIONABLE INSIGHTS GENERATION
    # =========================================================================
    print("\n[STEP 5] Generating actionable insights...")
    
    # Identify actionable features from top 10
    top_10_features_list = consensus_features.head(10).index.tolist()
    actionable_dist_features = [
        f for f in top_10_features_list 
        if any(x in f.lower() for x in ['phone', 'social', 'youtube', 'gaming', 'distraction', 'sleep', 'exercise', 'study', 'breaks', 'coffee'])
    ]
    
    print(f"\nTop actionable features from Top 10:")
    for i, feat in enumerate(actionable_dist_features[:5], 1):
        corr_val = leakage_results['corr_series'].get(feat, 0)
        print(f"  {i}. {feat} (correlation: {corr_val:+.4f})")
    
    # =========================================================================
    # STEP 6: FINAL SUMMARY STATISTICS
    # =========================================================================
    print("\n[STEP 6] Generating final summary statistics...")
    
    summary_stats = pd.DataFrame({
        'Metric': [
            'Total Samples',
            'Total Features',
            'Target Mean',
            'Target Std',
            'Target Min',
            'Target Max',
            'Baseline Model R²',
            'Top 10 Features R²',
            'R² Reduction (%)',
            'Features Removed',
        ],
        'Value': [
            len(df),
            len(model_cols),
            f"{y.mean():.4f}",
            f"{y.std():.4f}",
            f"{y.min():.2f}",
            f"{y.max():.2f}",
            f"{out_baseline['results_df'].iloc[0]['R2']:.6f}",
            f"{subset_results.iloc[0]['R2']:.6f}" if len(subset_results) > 0 else "N/A",
            f"{((out_baseline['results_df'].iloc[0]['R2'] - subset_results.iloc[0]['R2']) / out_baseline['results_df'].iloc[0]['R2'] * 100):.2f}%" if len(subset_results) > 0 else "N/A",
            f"{len(model_cols) - 10}",
        ]
    })
    
    print("\nFinal Summary Statistics:")
    print(summary_stats.to_string(index=False))
    summary_stats.to_csv(f"{OUTPUT_DIR}/optimization_summary_statistics.csv", index=False)
    
    print("\n" + "="*80)
    print("OPTIMIZATION PIPELINE COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("\nKey Generated Files:")
    print("  - leakage_investigation_report.txt")
    print("  - feature_target_correlations.csv / .png")
    print("  - mutual_information_scores.csv / .png")
    print("  - vif_analysis.csv / .png")
    print("  - f_statistic_analysis.csv")
    print("  - lasso_feature_selection.csv / .png")
    print("  - feature_selection_consensus.csv / .png")
    print("  - feature_subset_optimization_results.csv")
    print("  - feature_subset_comparison.png")
    print("  - shap_summary.png")
    print("  - shap_dependence_plots.png")
    print("  - optimization_summary_statistics.csv")
    
    return {
        'df': df,
        'X': X,
        'y': y,
        'model_cols': model_cols,
        'baseline_results': out_baseline,
        'leakage_results': leakage_results,
        'subset_results': subset_results,
        'consensus_features': consensus_features,
    }


if __name__ == "__main__":
    results = main()
