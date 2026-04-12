"""
Feature engineering for the Smart Student Productivity Advisor.
Adds composite and interaction features; handles leakage considerations.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .utils import TARGET_COLUMN


def add_digital_distraction_score(df, copy=True):
    """
    digital_distraction_score = phone + social_media + youtube + gaming (hours).
    Sum of all digital distraction hours per day.
    """
    if copy:
        df = df.copy()
    cols = ["phone_usage_hours", "social_media_hours", "youtube_hours", "gaming_hours"]
    df["digital_distraction_score"] = df[cols].sum(axis=1)
    return df


def add_healthy_lifestyle_score(df, copy=True):
    """
    healthy_lifestyle_score: combines sleep_hours and exercise_minutes.
    We normalize both to 0-1 scale so exercise_minutes does not dominate (e.g. 0-10 for sleep, 0-60 for exercise).
    Formula: mean of normalized sleep (by 0-12) and normalized exercise (by 0-120).
    """
    if copy:
        df = df.copy()
    sleep_n = np.clip(df["sleep_hours"] / 12.0, 0, 1)
    exercise_n = np.clip(df["exercise_minutes"] / 120.0, 0, 1)
    df["healthy_lifestyle_score"] = (sleep_n + exercise_n) / 2
    return df


def add_academic_engagement_score(df, copy=True):
    """
    academic_engagement_score = study_hours + normalized attendance + normalized assignments.
    Scale study (e.g. 0-12), attendance (0-100), assignments (e.g. 0-20) to comparable scale then sum or average.
    Using normalized (0-1) versions then mean so no single metric dominates.
    """
    if copy:
        df = df.copy()
    study_n = np.clip(df["study_hours_per_day"] / 12.0, 0, 1)
    att_n = np.clip(df["attendance_percentage"] / 100.0, 0, 1)
    assign_n = np.clip(df["assignments_completed"] / 20.0, 0, 1)
    df["academic_engagement_score"] = (study_n + att_n + assign_n) / 3
    return df


def add_study_to_phone_ratio(df, copy=True):
    """
    study_to_phone_ratio = study_hours_per_day / (phone_usage_hours + 1).
    Safe division; higher ratio means more study relative to phone use.
    """
    if copy:
        df = df.copy()
    df["study_to_phone_ratio"] = df["study_hours_per_day"] / (df["phone_usage_hours"] + 1)
    return df


def add_stress_focus_balance(df, copy=True):
    """
    stress_focus_balance = focus_score - stress_level.
    Positive when focus exceeds stress; negative otherwise.
    """
    if copy:
        df = df.copy()
    df["stress_focus_balance"] = df["focus_score"] - df["stress_level"]
    return df


def add_break_efficiency_score(df, copy=True):
    """
    break_efficiency_score: combines breaks_per_day with study_hours_per_day.
    Idea: moderate breaks with good study hours might be efficient. Simple: breaks / (study + 1) then normalized,
    or we use a score like: more breaks with more study = reasonable. Here we use (breaks / 10) * min(study/8, 1)
    so that we reward study and moderate breaks.
    """
    if copy:
        df = df.copy()
    break_n = np.clip(df["breaks_per_day"] / 10.0, 0, 1)
    study_contrib = np.clip(df["study_hours_per_day"] / 8.0, 0, 1)
    df["break_efficiency_score"] = break_n * study_contrib
    return df


def add_caffeine_stress_interaction(df, copy=True):
    """
    caffeine_stress_interaction = coffee_intake_mg * stress_level.
    Captures interaction between caffeine consumption and stress.
    """
    if copy:
        df = df.copy()
    df["caffeine_stress_interaction"] = df["coffee_intake_mg"] * df["stress_level"]
    return df


def add_grade_productivity_gap(df, copy=True):
    """
    grade_productivity_gap = final_grade - productivity_score.
    NOTE: For predicting productivity_score, using final_grade or grade_productivity_gap in features
    can cause leakage (final_grade is often highly correlated with productivity). We add this feature
    for EDA and optional analysis, but the modeling pipeline should exclude it (and optionally final_grade)
    when predicting productivity_score.
    """
    if copy:
        df = df.copy()
    df["grade_productivity_gap"] = df["final_grade"] - df["productivity_score"]
    return df


def engineer_all_features(df, copy=True, include_leaky=False):
    """
    Add all engineered features to the dataset.
    include_leaky: if True, add grade_productivity_gap and keep final_grade in feature set (for EDA).
    For modeling, use include_leaky=False and optionally drop final_grade in the model step.
    """
    if copy:
        df = df.copy()
    df = add_digital_distraction_score(df, copy=False)
    df = add_healthy_lifestyle_score(df, copy=False)
    df = add_academic_engagement_score(df, copy=False)
    df = add_study_to_phone_ratio(df, copy=False)
    df = add_stress_focus_balance(df, copy=False)
    df = add_break_efficiency_score(df, copy=False)
    df = add_caffeine_stress_interaction(df, copy=False)
    df = add_grade_productivity_gap(df, copy=False)
    return df


# Columns that may cause leakage when predicting productivity_score (final_grade is outcome-related)
LEAKY_OR_SENSITIVE_FOR_PREDICTION = ["final_grade", "grade_productivity_gap"]


def update_engineered_in_profile(profile_dict):
    """
    Update engineered feature values in a profile dict from its raw/base fields.
    Modifies only engineered keys; other keys are left as-is. Use after changing
    controllable features in a scenario so the model sees consistent engineered values.
    """
    p = profile_dict
    # Digital distraction score
    p["digital_distraction_score"] = (
        p.get("phone_usage_hours", 0) + p.get("social_media_hours", 0)
        + p.get("youtube_hours", 0) + p.get("gaming_hours", 0)
    )
    # Healthy lifestyle (normalized)
    sleep_n = np.clip(p.get("sleep_hours", 0) / 12.0, 0, 1)
    exercise_n = np.clip(p.get("exercise_minutes", 0) / 120.0, 0, 1)
    p["healthy_lifestyle_score"] = (sleep_n + exercise_n) / 2
    # Academic engagement (normalized)
    study_n = np.clip(p.get("study_hours_per_day", 0) / 12.0, 0, 1)
    att_n = np.clip(p.get("attendance_percentage", 0) / 100.0, 0, 1)
    assign_n = np.clip(p.get("assignments_completed", 0) / 20.0, 0, 1)
    p["academic_engagement_score"] = (study_n + att_n + assign_n) / 3
    # Study to phone ratio
    p["study_to_phone_ratio"] = p.get("study_hours_per_day", 0) / (p.get("phone_usage_hours", 0) + 1)
    # Stress-focus balance
    p["stress_focus_balance"] = p.get("focus_score", 0) - p.get("stress_level", 0)
    # Break efficiency
    break_n = np.clip(p.get("breaks_per_day", 0) / 10.0, 0, 1)
    study_contrib = np.clip(p.get("study_hours_per_day", 0) / 8.0, 0, 1)
    p["break_efficiency_score"] = break_n * study_contrib
    # Caffeine-stress interaction
    p["caffeine_stress_interaction"] = p.get("coffee_intake_mg", 0) * p.get("stress_level", 0)
    # Grade-productivity gap (use 0 if not available to avoid leakage)
    fg = p.get("final_grade", 0)
    ps = p.get("productivity_score", 0)
    p["grade_productivity_gap"] = fg - ps
    return p


def compute_engineered_from_profile(profile_dict):
    """
    Given a dict with base columns (e.g. study_hours_per_day, sleep_hours, ...),
    compute engineered features and return a new dict with all base + engineered keys.
    Used by the recommendation engine to keep features consistent when simulating scenarios.
    """
    from .utils import EXPECTED_COLUMNS
    base = [c for c in EXPECTED_COLUMNS if c != "student_id"]
    d = {k: profile_dict.get(k, 0) for k in base}
    df = pd.DataFrame([d])
    if "gender" in df.columns and "gender_encoded" not in df.columns:
        from .data_preprocessing import encode_gender
        df = encode_gender(df, copy=False)
        df = df.drop(columns=["gender"], errors="ignore")
    df = engineer_all_features(df, copy=False, include_leaky=True)
    return df.iloc[0].to_dict()


def get_feature_columns_for_model(include_final_grade=False, include_grade_gap=False):
    """
    Return list of feature names to use in the model.
    By default we exclude final_grade and grade_productivity_gap to avoid leakage.
    """
    # Base numeric features (after encoding) + engineered (excluding leaky)
    base = [
        "age", "gender_encoded",
        "study_hours_per_day", "sleep_hours", "phone_usage_hours", "social_media_hours",
        "youtube_hours", "gaming_hours", "breaks_per_day", "coffee_intake_mg", "exercise_minutes",
        "assignments_completed", "attendance_percentage", "stress_level", "focus_score",
        "digital_distraction_score", "healthy_lifestyle_score", "academic_engagement_score",
        "study_to_phone_ratio", "stress_focus_balance", "break_efficiency_score",
        "caffeine_stress_interaction",
    ]
    if include_final_grade:
        base.append("final_grade")
    if include_grade_gap:
        base.append("grade_productivity_gap")
    return base
