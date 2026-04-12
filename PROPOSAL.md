# DSA 210 — Project Proposal (Spring 2025–2026)

**Course:** DSA 210 — Introduction to Data Science  
**Term:** 2025–2026 Spring  
**Student:** İclal Yaren Hasaltın  
**Project title:** Smart Student Productivity Advisor: Measuring Digital Distraction and Recommending Productivity Improvements with Machine Learning

## Proposal (single-spaced; half–one page)

This project studies how **digital distraction** (phone, social media, YouTube, gaming) relates to student `**productivity_score`** and whether we can: (1) predict productivity from lifestyle and academic behavior features, and (2) recommend realistic lifestyle changes that improve predicted productivity.

Data source and collection: I will use a public tabular CSV dataset (or a compatible dataset aligned to the same schema) with one row per student and the required columns: `student_id, age, gender, study_hours_per_day, sleep_hours, phone_usage_hours, social_media_hours, youtube_hours, gaming_hours, breaks_per_day, coffee_intake_mg, exercise_minutes, assignments_completed, attendance_percentage, stress_level, focus_score, final_grade, productivity_score`. If some columns are missing, I will document the exact source(s) and explain how missing data are handled (transparent imputation vs. row removal). If I use self-collected survey data, I will collect anonymously with consent and anonymization.

Planned enrichment and leakage handling: I will engineer course-relevant features such as `digital_distraction_score`, `healthy_lifestyle_score`, `academic_engagement_score`, `study_to_phone_ratio`, `stress_focus_balance`, `break_efficiency_score`, and `caffeine_stress_interaction`. Because `final_grade` can be strongly related to productivity and may cause leakage, the main predictive model will exclude `final_grade` (and `grade_productivity_gap`).

Planned pipeline and evaluation: the workflow is loading + schema checks, missing/duplicate handling, IQR outlier removal, gender encoding, feature engineering, regression model training (Linear/Ridge/Lasso, Random Forest, Gradient Boosting, optionally XGBoost), and evaluation with **RMSE, MAE, and R²**. I will include EDA and diagnostics (correlation heatmap, actual vs. predicted, residuals), a digital-distraction interpretation section using feature importance (plus SHAP/permutation importance when available), a reduced distraction-only experiment vs. the full model, and a scenario-based recommendation engine that ranks realistic improvements by predicted gain.

Ethics and AI disclosure: I will not use `student_id` as a predictive feature and I will discuss that results show association rather than causation. AI assistance will be disclosed as required (used for drafting and implementation/debugging, with verification by running the code and checking outputs).