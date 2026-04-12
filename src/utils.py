"""
Shared utilities, constants, and configuration for the Smart Student Productivity Advisor project.
"""

import os

# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------
# Change this if your CSV file has a different name or path
DATA_PATH = "student_productivity.csv"

# Expected columns in the dataset (required for validation)
EXPECTED_COLUMNS = [
    "student_id",
    "age",
    "gender",
    "study_hours_per_day",
    "sleep_hours",
    "phone_usage_hours",
    "social_media_hours",
    "youtube_hours",
    "gaming_hours",
    "breaks_per_day",
    "coffee_intake_mg",
    "exercise_minutes",
    "assignments_completed",
    "attendance_percentage",
    "stress_level",
    "focus_score",
    "final_grade",
    "productivity_score",
]

# Target variable for prediction
TARGET_COLUMN = "productivity_score"

# Identifier column (not used as a predictive feature)
ID_COLUMN = "student_id"

# Digital distraction feature names (emphasized in analysis)
DIGITAL_DISTRACTION_FEATURES = [
    "phone_usage_hours",
    "social_media_hours",
    "youtube_hours",
    "gaming_hours",
]

# Positive lifestyle / academic features
POSITIVE_LIFESTYLE_FEATURES = [
    "study_hours_per_day",
    "sleep_hours",
    "exercise_minutes",
    "assignments_completed",
    "attendance_percentage",
    "focus_score",
]

# Controllable features for the recommendation engine (realistic to change)
CONTROLLABLE_FEATURES = [
    "study_hours_per_day",
    "sleep_hours",
    "phone_usage_hours",
    "social_media_hours",
    "youtube_hours",
    "gaming_hours",
    "breaks_per_day",
    "coffee_intake_mg",
    "exercise_minutes",
]

# Output directory for plots and saved artifacts
OUTPUT_DIR = "outputs"
MODEL_ARTIFACT_PATH = os.path.join(OUTPUT_DIR, "best_model.joblib")
PREPROCESSOR_ARTIFACT_PATH = os.path.join(OUTPUT_DIR, "preprocessor.joblib")


def get_data_path(project_root=None):
    """Return the absolute path to the dataset. Searches from project_root or cwd."""
    if project_root is None:
        project_root = os.getcwd()
    path = os.path.join(project_root, DATA_PATH)
    return os.path.abspath(path)


def ensure_output_dir():
    """Create output directory if it does not exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def validate_columns(df):
    """
    Check that the DataFrame has all expected columns.
    Raises ValueError with a friendly message if any are missing.
    """
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Expected columns: {EXPECTED_COLUMNS}"
        )
    return True
