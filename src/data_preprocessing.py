"""Data load + cleaning utils."""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from .utils import (
    DATA_PATH,
    EXPECTED_COLUMNS,
    TARGET_COLUMN,
    ID_COLUMN,
    get_data_path,
    validate_columns,
)


def generate_synthetic_data(n_samples=500, seed=42):
    """Make fake data if csv not found."""
    rng = np.random.default_rng(seed)
    n = n_samples

    age = rng.integers(18, 30, size=n)
    gender = rng.choice(["Male", "Female"], size=n)
    study_hours = np.clip(rng.normal(4, 1.5, n), 0.5, 12)
    sleep_hours = np.clip(rng.normal(7, 1.2, n), 4, 11)
    phone_usage = np.clip(rng.gamma(2, 1, n), 0, 8)
    social_media = np.clip(rng.gamma(1.5, 0.8, n), 0, 6)
    youtube_hours = np.clip(rng.gamma(1.2, 0.7, n), 0, 5)
    gaming_hours = np.clip(rng.gamma(1, 0.5, n), 0, 6)
    breaks_per_day = rng.integers(2, 12, size=n)
    coffee_mg = rng.integers(0, 400, size=n)
    exercise_min = np.clip(rng.gamma(20, 1.5, n), 0, 120).astype(int)
    assignments = rng.integers(0, 20, size=n)
    attendance_pct = np.clip(rng.normal(85, 10, n), 50, 100)
    stress_level = np.clip(rng.normal(5, 2, n), 1, 10)
    focus_score = np.clip(rng.normal(5, 2, n), 1, 10)

    # rough productivity formula, not perfect
    productivity = (
        30
        + 3 * study_hours
        + 2 * sleep_hours
        + 0.15 * exercise_min
        + 4 * focus_score
        + 0.2 * attendance_pct
        - 2 * phone_usage
        - 1.5 * social_media
        - 1 * youtube_hours
        - 1.2 * gaming_hours
        - 0.5 * stress_level
        + rng.normal(0, 5, n)
    )
    productivity = np.clip(productivity, 0, 100)

    final_grade = np.clip(productivity * 0.9 + rng.normal(0, 5, n), 0, 100)

    df = pd.DataFrame(
        {
            "student_id": [f"S{i:04d}" for i in range(n)],
            "age": age,
            "gender": gender,
            "study_hours_per_day": study_hours,
            "sleep_hours": sleep_hours,
            "phone_usage_hours": phone_usage,
            "social_media_hours": social_media,
            "youtube_hours": youtube_hours,
            "gaming_hours": gaming_hours,
            "breaks_per_day": breaks_per_day,
            "coffee_intake_mg": coffee_mg,
            "exercise_minutes": exercise_min,
            "assignments_completed": assignments,
            "attendance_percentage": attendance_pct,
            "stress_level": stress_level,
            "focus_score": focus_score,
            "final_grade": final_grade,
            "productivity_score": productivity,
        }
    )
    return df


def load_data(data_path=None, project_root=None, generate_if_missing=True):
    """Load csv; if missing can fallback to synthetic."""
    if data_path is None:
        data_path = get_data_path(project_root)

    if not os.path.isfile(data_path):
        if generate_if_missing:
            warnings.warn(
                f"Data file not found at {data_path}. Using synthetic data for demonstration."
            )
            return generate_synthetic_data(), True
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError("Dataset is empty.")
    return df, False


def basic_checks(df):
    """Run basic checks and print summary."""
    validate_columns(df)
    print("Dataset shape:", df.shape)
    print("Column names:", list(df.columns))
    print("Data types:\n", df.dtypes)
    print("\nFirst rows:\n", df.head())
    missing = df.isnull().sum()
    if missing.any():
        print("\nMissing values:\n", missing[missing > 0])
    else:
        print("\nMissing values: None")
    n_dup = df.duplicated().sum()
    print("Duplicate rows:", n_dup)
    print("\nBasic statistics:\n", df.describe())
    if "gender" in df.columns:
        print("\nGender value counts:\n", df["gender"].value_counts())
    return df


def handle_missing_values(df, strategy="drop"):
    """Handle missing values with drop or fill."""
    if df.isnull().sum().sum() == 0:
        return df
    if strategy == "drop":
        df = df.dropna()
    else:
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ["object", "category"]:
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")
                else:
                    df[col] = df[col].fillna(df[col].median())
    return df


def encode_gender(df, copy=True):
    """Encode gender to number."""
    if copy:
        df = df.copy()
    if "gender" not in df.columns:
        return df
    # keep strings a bit normalized
    g = df["gender"].astype(str).str.strip().str.title()
    le = LabelEncoder()
    df["gender_encoded"] = le.fit_transform(g)
    return df


def remove_outliers_iqr(df, columns=None, factor=1.5, copy=True):
    """Drop rows outside IQR bounds."""
    if copy:
        df = df.copy()
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = [ID_COLUMN] if ID_COLUMN in df.columns else []
    if columns is None:
        columns = [c for c in numeric if c not in exclude]
    else:
        columns = [c for c in columns if c in df.columns]
    mask = pd.Series(True, index=df.index)
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - factor * iqr
        high = q3 + factor * iqr
        mask &= (df[col] >= low) & (df[col] <= high)
    return df[mask].reset_index(drop=True)


def get_preprocessing_pipeline(include_target=True, numeric_features=None, scale=True):
    """
    Return a simple preprocessing pipeline: select numeric features, optionally scale.
    Used for model training. Caller fits on train and transforms train/test.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler as SkScaler

    steps = []
    if scale:
        steps.append(("scaler", SkScaler()))
    if not steps:
        return None
    return Pipeline(steps)


def prepare_for_model(
    df,
    target_col=TARGET_COLUMN,
    drop_id=True,
    encode_gender_col=True,
  ):
    """
    Prepare DataFrame for modeling: drop ID, encode gender, ensure numeric.
    Returns X (features), y (target), and list of feature names.
    Does not drop final_grade here; feature engineering / model config decides that.
    """
    df = df.copy()
    if drop_id and ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])
    if encode_gender_col and "gender" in df.columns:
        df = encode_gender(df, copy=False)
        df = df.drop(columns=["gender"], errors="ignore")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in DataFrame")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    # Only numeric columns
    X = X.select_dtypes(include=[np.number])
    feature_names = X.columns.tolist()
    return X, y, feature_names
