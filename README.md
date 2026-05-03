# Smart Student Productivity Advisor

**Measuring Digital Distraction and Recommending Productivity Improvements with Machine Learning**

---

## Project Objective

This data science project studies how **digital distraction** affects student productivity and builds:

1. **Productivity prediction** – Regression models to predict `productivity_score` from lifestyle and academic behavior.
2. **Digital distraction impact analysis** – Which distraction features hurt productivity the most.
3. **Recommendation engine** – Suggests realistic lifestyle changes to improve predicted productivity.

**Main target variable:** `productivity_score`

---
## Project Proposal (short)
This project studies how **digital distraction** (phone, social media, YouTube, gaming) relates to student **`productivity_score`** and whether we can: (1) predict productivity from lifestyle and academic behavior features, and (2) recommend realistic lifestyle changes that improve predicted productivity.

**Data source and collection.** I will use a public tabular CSV dataset (or a compatible dataset aligned to the same schema) with one row per student and the required columns: `student_id, age, gender, study_hours_per_day, sleep_hours, phone_usage_hours, social_media_hours, youtube_hours, gaming_hours, breaks_per_day, coffee_intake_mg, exercise_minutes, assignments_completed, attendance_percentage, stress_level, focus_score, final_grade, productivity_score`. If some columns are missing, I will document the exact source(s) and explain how missing data are handled (transparent imputation vs. row removal). If I use self-collected survey data, I will collect anonymously with consent and anonymization.

**Planned enrichment and leakage handling.** I will engineer course-relevant features such as `digital_distraction_score`, `healthy_lifestyle_score`, `academic_engagement_score`, `study_to_phone_ratio`, `stress_focus_balance`, `break_efficiency_score`, and `caffeine_stress_interaction`. Because `final_grade` can be strongly related to productivity and may cause leakage, the main predictive model will exclude `final_grade` (and `grade_productivity_gap`).

**Planned pipeline and evaluation.** the workflow is loading + schema checks, missing/duplicate handling, IQR outlier removal, gender encoding, feature engineering, regression model training (Linear/Ridge/Lasso, Random Forest, Gradient Boosting, optionally XGBoost), and evaluation with **RMSE, MAE, and R²**. I will include EDA/diagnostics (correlation heatmap, actual vs. predicted, residuals), an interpretation section (feature importance; plus SHAP/permutation importance when available), a reduced distraction-only experiment vs. the full model, and a scenario-based recommendation engine ranking realistic improvements by predicted gain.

**Ethics and AI disclosure.** I will not use `student_id` as a predictive feature and I will discuss association vs. causation. AI assistance will be disclosed as required (drafting and implementation/debugging, with verification by running the code and checking outputs).

---

## Dataset Description

The dataset contains student lifestyle, academic behavior, digital distraction, and productivity variables.

**Schema:**  
`student_id`, `age`, `gender`, `study_hours_per_day`, `sleep_hours`, `phone_usage_hours`, `social_media_hours`, `youtube_hours`, `gaming_hours`, `breaks_per_day`, `coffee_intake_mg`, `exercise_minutes`, `assignments_completed`, `attendance_percentage`, `stress_level`, `focus_score`, `final_grade`, `productivity_score`

- **Digital distraction:** phone_usage_hours, social_media_hours, youtube_hours, gaming_hours  
- **Positive factors:** study_hours_per_day, sleep_hours, exercise_minutes, assignments_completed, attendance_percentage, focus_score  

Place your CSV file as `student_productivity.csv` in the project root, or the code will generate synthetic data for demonstration.

---

## Methods Used

- **Data:** Loading, validation, missing/duplicate handling, basic statistics  
- **EDA:** Univariate/bivariate analysis, pairplot sample, missing-value profile, correlation heatmap, digital distraction vs productivity, hypothesis-style correlation tests  
- **Preprocessing:** Outlier handling (IQR), scaling, gender encoding  
- **Feature engineering:** digital_distraction_score, healthy_lifestyle_score, academic_engagement_score, study_to_phone_ratio, stress_focus_balance, break_efficiency_score, caffeine_stress_interaction, grade_productivity_gap  
- **Modeling:** Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting; optional XGBoost  
- **Evaluation:** RMSE, MAE, R² + cross-validation stats; model comparison; actual vs predicted; residual diagnostics; feature and permutation importance  
- **Digital distraction analysis:** Correlation, reduced-model experiment, interpretation  
- **Recommendation engine:** Scenario-based suggestions (reduce distraction, increase sleep/exercise/study) with realistic bounds

---

## Milestone 2 Checklist

This repository now covers the Milestone 2 requirement to **apply machine learning methods on the dataset**:

- Regression pipeline for predicting `productivity_score`
- Multiple model comparison with train/test evaluation
- Cross-validation summary in `outputs/model_comparison_table.csv`
- Saved artifacts for the best model in `outputs/best_model.joblib` and `outputs/preprocessor.joblib`
- Prediction output in `outputs/test_predictions.csv`
- Evaluation visuals such as actual-vs-predicted, residual plots, and feature/permutation importance
- A scenario-based recommendation engine built on top of the trained model

---

## How to Run

1. **Install dependencies** (use a virtual environment if needed)
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   Optional extras for heavier interpretation/experiments:
   ```bash
   pip install xgboost shap
   ```

2. **Prepare data**  
   Put your CSV at `student_productivity.csv` in the project root, or leave it missing to use generated synthetic data.

3. **Run the full pipeline**
   ```bash
   python main.py
   ```
   Or from the project root:
   ```bash
   cd dsa210-project-iclalyarenhasaltin
   python main.py
   ```

   Optional flags:
   ```bash
   python main.py --with-xgboost --with-shap
   ```

   The default run is intentionally lighter and more reproducible for milestone submission. XGBoost and SHAP are available as opt-in extras.

4. **Outputs**
   - Console: data summary, EDA insights, model comparison, recommendation examples
   - CSV in `outputs/`: `model_comparison_table.csv`, `test_predictions.csv`
   - Plots in `outputs/` (examples): distributions, boxplots, missing profile, pairplot sample, correlation heatmap, scatter plots, actual-vs-predicted, residual diagnostics, feature/permutation importance, recommendation improvement

---

## Project Structure

```
dsa210-project-iclalyarenhasaltin/
├── main.py
├── student_productivity.csv            # your data (optional; synthetic fallback exists)
├── PROPOSAL.md
├── requirements.txt
├── README.md
├── notebooks/
│   ├── 01_eda_and_hypothesis_tests.ipynb
│   └── 02_model_experiments_and_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── recommendation_engine.py
│   └── evaluation.py
└── outputs/                            # generated plots/tables/model artifacts
```

---

## Report Summary (for Course Report)

- **Problem definition:** Predict productivity and recommend lifestyle changes to improve it, with focus on digital distraction.  
- **Data preprocessing:** Schema validation, missing/duplicate handling, encoding, outlier treatment, scaling.  
- **Feature engineering:** Composite scores (digital distraction, healthy lifestyle, academic engagement), ratios and balances, interaction terms.  
- **Modeling:** Multiple regressors compared; best model selected by R²/RMSE/MAE. Optional heavier methods can be enabled from the CLI.  
- **Digital distraction analysis:** Correlation and reduced-model experiments show which distractions matter most.  
- **Recommendation engine:** Scenario simulation with realistic bounds; outputs ranked suggestions and expected gains.  
- **Evaluation:** Metrics table, residual and actual vs predicted plots, feature importance.  
- **Future improvements:** More data, temporal features, classification for “low/medium/high” productivity, A/B testing of recommendations.
