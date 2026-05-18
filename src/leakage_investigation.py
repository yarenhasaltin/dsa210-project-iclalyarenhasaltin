"""
Detailed leakage investigation and feature selection optimization.
Aims to lower R² by removing overly predictive/potentially leaky features 
and focusing on truly interpretable, actionable features.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    mutual_info_regression,
    SelectKBest,
    f_regression,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .utils import OUTPUT_DIR, TARGET_COLUMN, ensure_output_dir


def analyze_feature_target_correlation(X, y, feature_names):
    """Detailed correlation analysis between features and target."""
    corr_dict = {}
    for col in feature_names:
        corr_dict[col] = X[col].corr(y)
    
    corr_series = pd.Series(corr_dict).sort_values(ascending=False)
    
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS: Features vs Productivity_Score")
    print("="*70)
    print("\nHighest positive correlations:")
    print(corr_series.head(10).to_string())
    print("\nHighest negative correlations:")
    print(corr_series.tail(10).to_string())
    
    # Save to CSV
    corr_df = pd.DataFrame({
        'Feature': corr_series.index,
        'Correlation': corr_series.values
    })
    corr_df.to_csv(f"{OUTPUT_DIR}/feature_target_correlations.csv", index=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_series.sort_values().plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Correlation with Productivity Score')
    ax.set_title('Feature-Target Correlations\n(Sorted)')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_target_correlations.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return corr_series


def compute_mutual_information(X, y, feature_names):
    """Calculate mutual information scores."""
    print("\n" + "="*70)
    print("MUTUAL INFORMATION ANALYSIS")
    print("="*70)
    
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)
    
    print("\nMutual Information Scores (top 15):")
    print(mi_series.head(15).to_string())
    
    # Save
    mi_df = pd.DataFrame({
        'Feature': mi_series.index,
        'MI_Score': mi_series.values
    })
    mi_df.to_csv(f"{OUTPUT_DIR}/mutual_information_scores.csv", index=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    mi_series.head(20).sort_values().plot(kind='barh', ax=ax, color='coral')
    ax.set_xlabel('Mutual Information Score')
    ax.set_title('Feature Mutual Information Scores\n(Top 20 Features)')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/mutual_information_scores.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return mi_series


def compute_vif(X, feature_names):
    """Calculate Variance Inflation Factor to detect multicollinearity."""
    print("\n" + "="*70)
    print("MULTICOLLINEARITY ANALYSIS: Variance Inflation Factor (VIF)")
    print("="*70)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
    
    vif_data = pd.DataFrame({
        'Feature': feature_names,
        'VIF': [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
    }).sort_values('VIF', ascending=False)
    
    print("\nVIF Scores (VIF > 10 indicates high multicollinearity):")
    print(vif_data.to_string(index=False))
    
    # Save
    vif_data.to_csv(f"{OUTPUT_DIR}/vif_analysis.csv", index=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    vif_data_sorted = vif_data.sort_values('VIF')
    ax.barh(vif_data_sorted['Feature'], vif_data_sorted['VIF'], color='mediumvioletred')
    ax.axvline(x=10, color='red', linestyle='--', linewidth=2, label='VIF=10 (threshold)')
    ax.set_xlabel('VIF Score')
    ax.set_title('Variance Inflation Factor\n(Multicollinearity Detection)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/vif_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return vif_data


def compute_f_statistic(X, y, feature_names):
    """Compute F-statistic for each feature."""
    print("\n" + "="*70)
    print("F-STATISTIC ANALYSIS (Linear Regression Significance)")
    print("="*70)
    
    f_scores, p_values = f_regression(X, y)
    f_series = pd.Series(f_scores, index=feature_names).sort_values(ascending=False)
    p_series = pd.Series(p_values, index=feature_names)
    
    print("\nF-Statistic Scores (top 15):")
    f_df = pd.DataFrame({
        'Feature': f_series.index,
        'F_Score': f_series.values,
        'P_Value': [p_series[feat] for feat in f_series.index],
        'Significant': [p_series[feat] < 0.05 for feat in f_series.index]
    })
    print(f_df.to_string(index=False))
    
    # Save
    f_df.to_csv(f"{OUTPUT_DIR}/f_statistic_analysis.csv", index=False)
    
    return f_df


def lasso_feature_selection(X, y, feature_names):
    """Use LassoCV to identify important features (regularization-based selection)."""
    print("\n" + "="*70)
    print("LASSO FEATURE SELECTION")
    print("="*70)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lasso = LassoCV(cv=5, max_iter=10000, random_state=42)
    lasso.fit(X_scaled, y)
    
    coef_series = pd.Series(np.abs(lasso.coef_), index=feature_names).sort_values(ascending=False)
    
    print(f"\nLasso Alpha selected: {lasso.alpha_:.6f}")
    print("\nFeature Importance (absolute Lasso coefficients):")
    print(coef_series.to_string())
    
    # Count non-zero coefficients
    non_zero = (np.abs(lasso.coef_) > 1e-5).sum()
    print(f"\nNon-zero coefficients: {non_zero}/{len(feature_names)}")
    print(f"Selected features:\n{coef_series[coef_series > 1e-5].to_string()}")
    
    # Save
    coef_df = pd.DataFrame({
        'Feature': coef_series.index,
        'Lasso_Coefficient': coef_series.values
    })
    coef_df.to_csv(f"{OUTPUT_DIR}/lasso_feature_selection.csv", index=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    coef_series.head(20).sort_values().plot(kind='barh', ax=ax, color='teal')
    ax.set_xlabel('|Lasso Coefficient|')
    ax.set_title('Lasso Feature Selection\n(Top 20 Features by Coefficient Magnitude)')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/lasso_feature_selection.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    selected_features = coef_series[coef_series > 1e-5].index.tolist()
    return selected_features, coef_series


def permutation_importance_analysis(model, X_test, y_test, feature_names):
    """Compute permutation importance using the best trained model."""
    print("\n" + "="*70)
    print("PERMUTATION IMPORTANCE ANALYSIS")
    print("="*70)
    
    perm_imp = permutation_importance(
        model, X_test, y_test, 
        n_repeats=10, 
        random_state=42, 
        n_jobs=-1
    )
    
    perm_series = pd.Series(
        perm_imp.importances_mean, 
        index=feature_names
    ).sort_values(ascending=False)
    
    print("\nPermutation Importance (top 15):")
    perm_df = pd.DataFrame({
        'Feature': perm_series.index,
        'Importance_Mean': perm_series.values,
        'Importance_Std': [perm_imp.importances_std[i] for i, f in enumerate(feature_names) if f in perm_series.index]
    })
    print(perm_df.head(15).to_string(index=False))
    
    # Save
    perm_df.to_csv(f"{OUTPUT_DIR}/permutation_importance_analysis.csv", index=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    perm_series.head(20).sort_values().plot(kind='barh', ax=ax, color='forestgreen')
    ax.set_xlabel('Permutation Importance')
    ax.set_title('Permutation Importance Analysis\n(Top 20 Features)')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/permutation_importance_sorted.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return perm_series


def summarize_feature_selection_methods(
    corr_series, mi_series, f_df, lasso_selected, perm_series
):
    """
    Summarize all feature selection methods and create consensus ranking.
    Features that consistently rank high across methods are most reliable.
    """
    print("\n" + "="*70)
    print("FEATURE SELECTION CONSENSUS RANKING")
    print("="*70)
    
    # Normalize and rank each method
    consensus = pd.DataFrame()
    
    consensus['Feature'] = corr_series.index
    
    # Correlation ranking
    corr_rank = pd.Series(range(len(corr_series), 0, -1), index=corr_series.index)
    
    # MI ranking
    mi_rank = pd.Series(range(len(mi_series), 0, -1), index=mi_series.index)
    
    # F-statistic ranking
    f_rank = pd.Series(
        range(len(f_df), 0, -1),
        index=f_df.sort_values('F_Score', ascending=False)['Feature'].values
    )
    
    # Permutation importance ranking
    perm_rank = pd.Series(range(len(perm_series), 0, -1), index=perm_series.index)
    
    # Compute average rank
    features = set(corr_series.index)
    avg_ranks = {}
    for feat in features:
        ranks = [
            corr_rank.get(feat, len(features)),
            mi_rank.get(feat, len(features)),
            f_rank.get(feat, len(features)),
            perm_rank.get(feat, len(features))
        ]
        avg_ranks[feat] = np.mean(ranks)
    
    consensus_series = pd.Series(avg_ranks).sort_values()
    
    print("\nConsensus Ranking (lower = more important):")
    print("Top 20 Most Important Features (by consensus):")
    consensus_df = pd.DataFrame({
        'Feature': consensus_series.index,
        'Avg_Rank': consensus_series.values
    }).head(20)
    print(consensus_df.to_string(index=False))
    
    # Save
    consensus_df.to_csv(f"{OUTPUT_DIR}/feature_selection_consensus.csv", index=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    consensus_series.head(20).sort_values(ascending=False).plot(kind='barh', ax=ax, color='purple')
    ax.set_xlabel('Consensus Rank (Lower = More Important)')
    ax.set_title('Feature Selection Consensus Ranking\n(Aggregated from 4 Methods)')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_selection_consensus.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return consensus_series


def generate_leakage_report(
    X, y, feature_names, corr_series, vif_data, f_df, 
    lasso_selected, perm_series, consensus_series
):
    """Generate a comprehensive leakage and feature selection report."""
    
    report = []
    report.append("="*80)
    report.append("LEAKAGE INVESTIGATION AND FEATURE SELECTION REPORT")
    report.append("="*80)
    report.append("")
    
    # High correlations
    very_high_corr = corr_series[np.abs(corr_series) > 0.95]
    report.append("POTENTIAL LEAKAGE INDICATORS:")
    report.append("-" * 80)
    if len(very_high_corr) > 0:
        report.append(f"\n⚠️  ALERT: Found {len(very_high_corr)} features with |correlation| > 0.95:")
        for feat, corr_val in very_high_corr.items():
            report.append(f"   - {feat}: {corr_val:.6f}")
        report.append("\n   These features may represent near-perfect proxies or direct")
        report.append("   calculations of the target. Consider removing for more realistic models.")
    else:
        report.append("\n✓ No features with |correlation| > 0.95 detected.")
    report.append("")
    
    # Multicollinearity
    report.append("MULTICOLLINEARITY ANALYSIS:")
    report.append("-" * 80)
    high_vif = vif_data[vif_data['VIF'] > 10]
    if len(high_vif) > 0:
        report.append(f"\n⚠️  Found {len(high_vif)} features with VIF > 10 (high multicollinearity):")
        for _, row in high_vif.iterrows():
            report.append(f"   - {row['Feature']}: VIF = {row['VIF']:.2f}")
        report.append("\n   High VIF indicates features are linear combinations of others,")
        report.append("   suggesting engineered/derived features.")
    else:
        report.append("\n✓ No severe multicollinearity detected (all VIF < 10).")
    report.append("")
    
    # Statistical significance
    report.append("FEATURE SIGNIFICANCE:")
    report.append("-" * 80)
    sig_features = f_df[f_df['Significant']]
    report.append(f"\n{len(sig_features)}/{len(f_df)} features are statistically significant (p < 0.05)")
    insignificant = f_df[~f_df['Significant']]
    if len(insignificant) > 0:
        report.append(f"\nInsignificant features (candidates for removal):")
        for _, row in insignificant.iterrows():
            report.append(f"   - {row['Feature']}: p-value = {row['P_Value']:.4f}")
    report.append("")
    
    # Feature selection consensus
    report.append("RECOMMENDED FEATURE SUBSET (by consensus ranking):")
    report.append("-" * 80)
    top_10_features = consensus_series.head(10).index.tolist()
    report.append(f"\nTop 10 most important features (by consensus):")
    for i, feat in enumerate(top_10_features, 1):
        report.append(f"   {i}. {feat}")
    
    report.append(f"\n\nRecommended minimal feature set (top 10):")
    report.append("These features provide best balance of interpretability and predictiveness.")
    report.append("")
    
    # Lasso selection
    report.append("LASSO-SELECTED FEATURES (regularization-based):")
    report.append("-" * 80)
    report.append(f"\nLasso identified {len(lasso_selected)} essential features:")
    for feat in lasso_selected[:15]:
        report.append(f"   - {feat}")
    report.append("")
    
    # Dataset characteristics
    report.append("DATASET CHARACTERISTICS:")
    report.append("-" * 80)
    report.append(f"\nTotal features: {len(feature_names)}")
    report.append(f"Total samples: {len(X)}")
    report.append(f"Target variable: productivity_score")
    report.append(f"Target mean: {y.mean():.2f}, std: {y.std():.2f}")
    report.append(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS FOR IMPROVED MODEL:")
    report.append("-" * 80)
    report.append("""
1. FEATURE REDUCTION:
   - Use top 10-15 consensus-ranked features for more interpretable models
   - This reduces overfitting and improves real-world applicability
   
2. REMOVE REDUNDANT FEATURES:
   - Digital distraction score is sum of its components (removes 4 components)
   - Healthy lifestyle score is derived from sleep + exercise (consider using raw values)
   
3. FOCUS ON ACTIONABLE FEATURES:
   - Prioritize features that students can change (e.g., sleep, phone usage)
   - De-prioritize fixed features (age, gender, final_grade)
   
4. INTERPRET R² CAREFULLY:
   - Very high R² may indicate synthetic data or near-deterministic relationships
   - Use domain knowledge to validate if predictions make real-world sense
   
5. CROSS-VALIDATION:
   - Ensure cross-validation uses stratification by key variables
   - Test on truly unseen data to validate generalization
""")
    
    return "\n".join(report)


def run_leakage_investigation(X, y, feature_names, model=None, X_test=None, y_test=None):
    """Main orchestration function."""
    ensure_output_dir()
    
    print("\n" + "="*80)
    print("STARTING DETAILED LEAKAGE AND FEATURE SELECTION INVESTIGATION")
    print("="*80)
    
    # Step 1: Correlations
    corr_series = analyze_feature_target_correlation(X, y, feature_names)
    
    # Step 2: Mutual information
    mi_series = compute_mutual_information(X, y, feature_names)
    
    # Step 3: VIF
    vif_data = compute_vif(X, feature_names)
    
    # Step 4: F-statistics
    f_df = compute_f_statistic(X, y, feature_names)
    
    # Step 5: Lasso
    lasso_selected, lasso_coefs = lasso_feature_selection(X, y, feature_names)
    
    # Step 6: Permutation importance (if model is available)
    perm_series = None
    if model is not None and X_test is not None and y_test is not None:
        perm_series = permutation_importance_analysis(model, X_test, y_test, feature_names)
    
    # Step 7: Consensus
    consensus_series = summarize_feature_selection_methods(
        corr_series, mi_series, f_df, lasso_selected, 
        perm_series if perm_series is not None else pd.Series(0, index=feature_names)
    )
    
    # Step 8: Generate report
    report = generate_leakage_report(
        X, y, feature_names, corr_series, vif_data, f_df,
        lasso_selected, perm_series, consensus_series
    )
    
    # Save report
    with open(f"{OUTPUT_DIR}/leakage_investigation_report.txt", "w") as f:
        f.write(report)
    
    print("\n" + report)
    
    return {
        'corr_series': corr_series,
        'mi_series': mi_series,
        'vif_data': vif_data,
        'f_df': f_df,
        'lasso_selected': lasso_selected,
        'perm_series': perm_series,
        'consensus_series': consensus_series,
        'report': report
    }
