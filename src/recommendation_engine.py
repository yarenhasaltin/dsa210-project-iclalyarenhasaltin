"""
Recommendation engine: suggests lifestyle changes to improve predicted productivity.
Uses scenario simulation with realistic bounds.
"""

import numpy as np
import pandas as pd
from copy import deepcopy

from .utils import CONTROLLABLE_FEATURES
from .feature_engineering import update_engineered_in_profile


# Realistic bounds for controllable features (min, max)
FEATURE_BOUNDS = {
    "study_hours_per_day": (0.5, 12),
    "sleep_hours": (4, 10),
    "phone_usage_hours": (0, 24),
    "social_media_hours": (0, 24),
    "youtube_hours": (0, 24),
    "gaming_hours": (0, 24),
    "breaks_per_day": (0, 20),
    "coffee_intake_mg": (0, 500),
    "exercise_minutes": (0, 120),
}


def _clamp_profile(profile, bounds=None):
    """Clamp profile values to realistic bounds."""
    bounds = bounds or FEATURE_BOUNDS
    out = profile.copy()
    for k, (lo, hi) in bounds.items():
        if k in out and isinstance(out[k], (int, float)):
            out[k] = np.clip(out[k], lo, hi)
    return out


def _profile_to_row(profile, feature_names, fill_missing=0):
    """Convert a dict profile to a row DataFrame with the same columns as feature_names. Ensures engineered features are updated."""
    p = dict(profile)
    update_engineered_in_profile(p)
    row = {}
    for f in feature_names:
        row[f] = p.get(f, fill_missing)
    return pd.DataFrame([row])[feature_names]


def _predict_one(model, preprocessor, row_df, use_scaled):
    """Single prediction using model and preprocessor."""
    if use_scaled and preprocessor.get("scaler") is not None:
        X = preprocessor["scaler"].transform(row_df)
        X = pd.DataFrame(X, columns=row_df.columns)
    else:
        X = row_df
    return float(model.predict(X)[0])


def generate_scenarios(base_profile, feature_names, bounds=None):
    """
    Generate realistic improvement scenarios by modifying controllable features.
    Returns list of (description, modified_profile) where modified_profile is a dict.
    """
    bounds = bounds or FEATURE_BOUNDS
    scenarios = []
    profile = deepcopy(base_profile)

    # Reduce distractions (realistic decrements)
    for feat, delta in [
        ("phone_usage_hours", 2),
        ("phone_usage_hours", 1),
        ("social_media_hours", 2),
        ("social_media_hours", 1),
        ("youtube_hours", 2),
        ("youtube_hours", 1),
        ("gaming_hours", 2),
        ("gaming_hours", 1),
    ]:
        if feat not in profile:
            continue
        p = deepcopy(profile)
        p[feat] = max(bounds[feat][0], p[feat] - delta)
        if p[feat] != profile[feat]:
            scenarios.append((f"Reduce {feat.replace('_', ' ')} by {delta} hour(s)", p))

    # Increase sleep
    for delta in [2, 1.5, 1]:
        p = deepcopy(profile)
        p["sleep_hours"] = min(bounds["sleep_hours"][1], p.get("sleep_hours", 7) + delta)
        if p["sleep_hours"] != profile.get("sleep_hours"):
            scenarios.append((f"Increase sleep by {delta} hour(s)", p))

    # Increase exercise
    for delta in [30, 20, 10]:
        p = deepcopy(profile)
        p["exercise_minutes"] = min(bounds["exercise_minutes"][1], p.get("exercise_minutes", 0) + delta)
        if p["exercise_minutes"] != profile.get("exercise_minutes"):
            scenarios.append((f"Increase exercise by {delta} minutes", p))

    # Increase study
    for delta in [1.5, 1]:
        p = deepcopy(profile)
        p["study_hours_per_day"] = min(bounds["study_hours_per_day"][1], p.get("study_hours_per_day", 4) + delta)
        if p["study_hours_per_day"] != profile.get("study_hours_per_day"):
            scenarios.append((f"Increase study time by {delta} hour(s)", p))

    # Combined: reduce one distraction + improve one positive
    combos = [
        ("Reduce phone usage by 2 hours", "phone_usage_hours", 2, "and increase sleep by 1.5 hours", "sleep_hours", 1.5),
        ("Reduce phone usage by 2 hours", "phone_usage_hours", 2, "and increase exercise by 20 minutes", "exercise_minutes", 20),
        ("Reduce social media by 1 hour", "social_media_hours", 1, "and increase sleep by 1.5 hours", "sleep_hours", 1.5),
        ("Reduce social media by 1 hour", "social_media_hours", 1, "and increase exercise by 20 minutes", "exercise_minutes", 20),
        ("Reduce gaming by 1 hour", "gaming_hours", 1, "and increase study time by 1 hour", "study_hours_per_day", 1),
        ("Reduce youtube by 1 hour", "youtube_hours", 1, "and increase study time by 1 hour", "study_hours_per_day", 1),
    ]
    for desc_neg, feat_neg, delta_neg, desc_pos, pos_feat, pos_delta in combos:
        p = deepcopy(profile)
        p[feat_neg] = max(bounds[feat_neg][0], p.get(feat_neg, 0) - delta_neg)
        if pos_feat == "sleep_hours":
            p[pos_feat] = min(bounds[pos_feat][1], p.get(pos_feat, 7) + pos_delta)
        elif pos_feat == "exercise_minutes":
            p[pos_feat] = min(bounds[pos_feat][1], p.get(pos_feat, 0) + pos_delta)
        else:
            p[pos_feat] = min(bounds[pos_feat][1], p.get(pos_feat, 4) + pos_delta)
        p = _clamp_profile(p, bounds)
        scenarios.append((f"{desc_neg} {desc_pos}", p))

    # Deduplicate by description and clamp
    seen = set()
    unique_scenarios = []
    for desc, p in scenarios:
        p = _clamp_profile(p, bounds)
        controllable_tuple = tuple(
            (k, round(float(p.get(k, 0)), 2)) for k in sorted(p.keys()) if k in CONTROLLABLE_FEATURES
        )
        key = (desc, controllable_tuple)
        if key in seen:
            continue
        seen.add(key)
        unique_scenarios.append((desc, p))
    return unique_scenarios


def recommend_improvements(
    student_profile,
    trained_model,
    preprocessor,
    feature_names,
    top_k=5,
):
    """
    Main entry: get top recommendations for a student profile.

    student_profile: dict or Series with at least controllable + any other features needed by the model.
    trained_model: fitted regressor (or Pipeline).
    preprocessor: dict with 'scaler' and 'feature_names' (or 'feature_names' only), and 'best_uses_scaled'.
    feature_names: list of column names the model expects (same order as training).

    Returns:
        current_pred: float
        recommendations: list of dicts with keys description, new_score, improvement
    """
    # Ensure we have a dict and all required keys for the model
    if hasattr(student_profile, "to_dict"):
        profile = student_profile.to_dict()
    else:
        profile = dict(student_profile)
    for f in feature_names:
        if f not in profile and f in FEATURE_BOUNDS:
            profile[f] = 0
        elif f not in profile:
            profile[f] = 0

    use_scaled = preprocessor.get("best_uses_scaled", True)
    row = _profile_to_row(profile, feature_names)
    current_pred = _predict_one(trained_model, preprocessor, row, use_scaled)

    scenarios = generate_scenarios(profile, feature_names)
    results = []
    for desc, mod_profile in scenarios:
        mod_row = _profile_to_row(mod_profile, feature_names)
        new_score = _predict_one(trained_model, preprocessor, mod_row, use_scaled)
        improvement = new_score - current_pred
        results.append({
            "description": desc,
            "new_score": new_score,
            "improvement": improvement,
            "profile": mod_profile,
        })
    results.sort(key=lambda x: -x["improvement"])
    top = results[:top_k]
    return current_pred, [{"description": r["description"], "new_score": r["new_score"], "improvement": r["improvement"]} for r in top]


def format_recommendations_output(current_pred, recommendations):
    """Pretty-print current score and top recommendations."""
    lines = [f"Current predicted productivity: {current_pred:.1f}", "", "Top recommendations:"]
    for i, r in enumerate(recommendations, 1):
        lines.append(f"{i}. {r['description']}")
        lines.append(f"   New predicted productivity: {r['new_score']:.1f}")
        lines.append(f"   Improvement: {r['improvement']:+.1f}")
        lines.append("")
    return "\n".join(lines)
