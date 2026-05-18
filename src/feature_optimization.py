"""
Advanced model optimization with multiple feature subset comparisons
and interpretability focus.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .utils import OUTPUT_DIR, TARGET_COLUMN, ensure_output_dir

warnings.filterwarnings('ignore')


def train_model_with_features(X_train, X_test, y_train, y_test, feature_subset_name):
    """Train a fitted simple ensemble and return metrics."""
    
    # Use a simple but effective approach: Ridge + GB ensemble
    # Ridge for interpretability, GB for accuracy
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Simple ensemble: Ridge (for interpretability) 
    ridge = RidgeCV(alphas=np.logspace(-2, 3, 20), cv=5)
    ridge.fit(X_train_scaled, y_train)
    ridge_pred = ridge.predict(X_test_scaled)
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    
    # Ensemble prediction (simple average)
    ensemble_pred = (ridge_pred + gb_pred) / 2
    
    # Compute metrics
    metrics = {
        'Feature_Subset': feature_subset_name,
        'N_Features': X_train.shape[1],
        'RMSE': float(np.sqrt(mean_squared_error(y_test, ensemble_pred))),
        'MAE': float(mean_absolute_error(y_test, ensemble_pred)),
        'R2': float(r2_score(y_test, ensemble_pred)),
        'Ridge_R2': float(r2_score(y_test, ridge_pred)),
        'GB_R2': float(r2_score(y_test, gb_pred)),
    }
    
    return metrics, ensemble_pred, ridge, gb, X_test_scaled


def run_feature_subset_experiments(df, X, y, feature_names, consensus_features):
    """Test multiple feature subset scenarios to find optimal balance."""
    
    ensure_output_dir()
    
    print("\n" + "="*80)
    print("FEATURE SUBSET EXPERIMENTS: Comparing Different Feature Sets")
    print("="*80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    results_list = []
    
    # Scenario 1: All features (baseline)
    print("\n[1/5] Training on ALL features...")
    metrics_all, pred_all, ridge_all, gb_all, X_test_all = train_model_with_features(
        X_train, X_test, y_train, y_test, "All Features (All)"
    )
    results_list.append(metrics_all)
    print(f"   -> R² = {metrics_all['R2']:.6f}, RMSE = {metrics_all['RMSE']:.4f}")
    
    # Scenario 2: Top 10 consensus features
    print("\n[2/5] Training on TOP 10 consensus features...")
    top_10 = consensus_features.head(10).index.tolist()
    X_train_top10 = X_train[top_10]
    X_test_top10 = X_test[top_10]
    metrics_top10, pred_top10, ridge_top10, gb_top10, X_test_top10_scaled = train_model_with_features(
        X_train_top10, X_test_top10, y_train, y_test, "Top 10 Features"
    )
    results_list.append(metrics_top10)
    print(f"   -> R² = {metrics_top10['R2']:.6f}, RMSE = {metrics_top10['RMSE']:.4f}")
    
    # Scenario 3: Top 15 consensus features
    print("\n[3/5] Training on TOP 15 consensus features...")
    top_15 = consensus_features.head(15).index.tolist()
    X_train_top15 = X_train[top_15]
    X_test_top15 = X_test[top_15]
    metrics_top15, pred_top15, ridge_top15, gb_top15, X_test_top15_scaled = train_model_with_features(
        X_train_top15, X_test_top15, y_train, y_test, "Top 15 Features"
    )
    results_list.append(metrics_top15)
    print(f"   -> R² = {metrics_top15['R2']:.6f}, RMSE = {metrics_top15['RMSE']:.4f}")
    
    # Scenario 4: Actionable features (student can change)
    print("\n[4/5] Training on ACTIONABLE features (changeable by student)...")
    actionable_features = [
        'study_hours_per_day', 'sleep_hours', 'phone_usage_hours', 'social_media_hours',
        'youtube_hours', 'gaming_hours', 'breaks_per_day', 'coffee_intake_mg', 
        'exercise_minutes', 'digital_distraction_score', 'healthy_lifestyle_score',
        'study_to_phone_ratio', 'stress_focus_balance', 'break_efficiency_score'
    ]
    actionable_present = [f for f in actionable_features if f in feature_names]
    X_train_actionable = X_train[actionable_present]
    X_test_actionable = X_test[actionable_present]
    metrics_actionable, pred_actionable, ridge_actionable, gb_actionable, X_test_actionable_scaled = train_model_with_features(
        X_train_actionable, X_test_actionable, y_train, y_test, "Actionable Features"
    )
    results_list.append(metrics_actionable)
    print(f"   -> R² = {metrics_actionable['R2']:.6f}, RMSE = {metrics_actionable['RMSE']:.4f}")
    
    # Scenario 5: Minimal feature set (top 5)
    print("\n[5/5] Training on TOP 5 features (minimal set)...")
    top_5 = consensus_features.head(5).index.tolist()
    X_train_top5 = X_train[top_5]
    X_test_top5 = X_test[top_5]
    metrics_top5, pred_top5, ridge_top5, gb_top5, X_test_top5_scaled = train_model_with_features(
        X_train_top5, X_test_top5, y_train, y_test, "Top 5 Features"
    )
    results_list.append(metrics_top5)
    print(f"   -> R² = {metrics_top5['R2']:.6f}, RMSE = {metrics_top5['RMSE']:.4f}")
    
    # Create comparison dataframe
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('R2', ascending=False)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY (sorted by R²):")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Save
    results_df.to_csv(f"{OUTPUT_DIR}/feature_subset_experiments.csv", index=False)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # R² comparison
    results_sorted = results_df.sort_values('R2', ascending=True)
    axes[0].barh(results_sorted['Feature_Subset'], results_sorted['R2'], color='steelblue')
    axes[0].set_xlabel('R² Score')
    axes[0].set_title('R² Comparison Across Feature Subsets')
    axes[0].set_xlim([0, 1])
    
    # RMSE comparison
    results_sorted_rmse = results_df.sort_values('RMSE', ascending=True)
    axes[1].barh(results_sorted_rmse['Feature_Subset'], results_sorted_rmse['RMSE'], color='coral')
    axes[1].set_xlabel('RMSE')
    axes[1].set_title('RMSE Comparison Across Feature Subsets')
    
    # Number of features vs R²
    axes[2].scatter(results_df['N_Features'], results_df['R2'], s=200, alpha=0.7, color='purple')
    for _, row in results_df.iterrows():
        axes[2].annotate(row['Feature_Subset'], 
                        (row['N_Features'], row['R2']),
                        fontsize=8, ha='right')
    axes[2].set_xlabel('Number of Features')
    axes[2].set_ylabel('R² Score')
    axes[2].set_title('Feature Count vs Model R²')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_subset_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Calculate improvements
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    r2_all = metrics_all['R2']
    r2_top10 = metrics_top10['R2']
    r2_drop = (r2_all - r2_top10) / r2_all * 100 if r2_all != 0 else 0
    
    print(f"\nR² drop from All features -> Top 10: {r2_drop:.2f}%")
    print(f"  - Using {metrics_top10['N_Features']} features instead of {metrics_all['N_Features']}")
    print(f"  - {metrics_top10['N_Features']/metrics_all['N_Features']*100:.1f}% of original feature count")
    
    if r2_drop <= 5:
        print(f"\n✓ Top 10 features capture {100-r2_drop:.1f}% of model performance")
        print(f"  Recommendation: Use Top 10 for better interpretability with minimal accuracy loss")
    elif r2_drop <= 10:
        print(f"\n✓ Top 15 features recommended for balance between interpretability and accuracy")
    else:
        print(f"\n⚠  Significant R² drop with reduced features - full feature set may be optimal")
    
    return results_df, {
        'All': (X_train, X_test, pred_all),
        'Top10': (X_train_top10, X_test_top10, pred_top10),
        'Top15': (X_train_top15, X_test_top15, pred_top15),
        'Actionable': (X_train_actionable, X_test_actionable, pred_actionable),
        'Top5': (X_train_top5, X_test_top5, pred_top5),
    }


def generate_feature_selection_summary():
    """Generate summary document for feature selection findings."""
    
    summary = """
================================================================================
FEATURE SELECTION AND LEAKAGE INVESTIGATION FINDINGS
================================================================================

PROJECT CONTEXT:
The project aims to predict student productivity_score from behavioral and 
academic features, with a focus on understanding digital distraction impacts 
and providing actionable recommendations for improvement.

KEY INVESTIGATION QUESTIONS:
1. Are engineered features introducing leakage or overfitting?
2. Which features are truly predictive of student productivity?
3. Can we achieve reasonable accuracy with fewer, more interpretable features?
4. Which features are actionable (students can change them)?

LEAKAGE INVESTIGATION PROCESS:
1. Correlation Analysis - Identify features too correlated with target (|r| > 0.95)
2. Multicollinearity (VIF) - Detect redundant/derived features
3. Statistical Significance (F-test) - Identify truly important features
4. Mutual Information - Measure non-linear feature importance
5. Lasso Regularization - Determine essential feature subset
6. Consensus Ranking - Aggregate rankings from all methods

ENGINEERED FEATURES ANALYSIS:
The dataset includes engineered composite features:
- digital_distraction_score = sum of phone/social/youtube/gaming hours
  → This is a LEAKY feature (linear combination of base features)
  → Consider using component features instead for cleaner modeling
  
- healthy_lifestyle_score = normalized mean of sleep and exercise
  → Derived from base features, adds interpretability
  → Consider including raw features for flexibility
  
- academic_engagement_score = normalized mean of study/attendance/assignments
  → Similar interpretation as healthy_lifestyle_score
  
- study_to_phone_ratio, stress_focus_balance, break_efficiency_score
  → Interaction/ratio features, potentially valuable for interpretation

RECOMMENDATION HIERARCHY:

Tier 1 - Use these features:
(Top consensus-ranked, statistically significant, low multicollinearity)
  - Selected from feature selection consensus analysis showing consistent 
    importance across all methods

Tier 2 - Consider these features:
(Moderate importance, some multicollinearity, provide additional context)
  - Use when interpretability/explainability is less critical

Tier 3 - Minimize use of:
(High multicollinearity, engineered from Tier 1 features)
  - Can be recomputed from Tier 1 if needed
  - Usually sum/mean/ratio of Tier 1 features

Tier 4 - Remove from predictive models:
  - student_id (no information value)
  - final_grade (potential target leakage)
  - grade_productivity_gap (derived from target)

ACTIONABILITY PRINCIPLE:
Features should be prioritized based on student ability to change them:

    Highly Actionable (students can directly control):
    ✓ study_hours_per_day
    ✓ phone_usage_hours, social_media_hours, youtube_hours, gaming_hours
    ✓ sleep_hours
    ✓ exercise_minutes
    ✓ breaks_per_day
    ✓ coffee_intake_mg

    Partially Actionable (depends on context):
    ~ stress_level (can be reduced through various strategies)
    ~ focus_score (can be improved with techniques)
    ~ assignments_completed (somewhat controllable)
    
    Not Actionable (fixed or dependent):
    ✗ age, gender
    ✗ attendance_percentage (often required)
    ✗ final_grade (outcome variable, not a predictor)

MODEL PERFORMANCE TARGETS:

R² Interpretation:
- R² > 0.99: Suspicious, likely indicates leakage or synthetic data
- R² 0.85-0.98: Very good (but validate with domain experts)
- R² 0.70-0.85: Good (reasonable for behavioral prediction)
- R² 0.50-0.70: Moderate (acceptable with large error bars)
- R² < 0.50: Poor (model not useful)

Given current R² ≈ 0.9999999, recommend:
1. Focus on feature subset experiments to find "realistic" performance level
2. Emphasize uncertainty in report (confidence intervals, cross-validation)
3. Validate with domain experts whether relationships make real-world sense
4. Be conservative in recommendations to users

ACTIONABLE INSIGHTS GENERATION:

Once feature importance is determined, generate recommendations by:
1. Identifying top 5-10 most impactful features
2. For each feature, simulate realistic changes (±1 SD)
3. Predict productivity improvement
4. Rank by feasibility & expected impact
5. Present to student with confidence intervals

Example: "If you reduce phone usage from 5 to 3 hours/day, predicted 
productivity could improve from 45 to 52 (±3), based on this model."

================================================================================
"""
    
    return summary
