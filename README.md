# Credit Risk Prediction — Loan Default Classification

Predicts whether a loan applicant is likely to default, using applicant demographics and loan characteristics, to support lending/underwriting decisions.

> **Data note:** This repo currently runs on a synthetically generated dataset built to match the schema of the public Kaggle *Credit Risk Dataset* (`person_age`, `person_income`, `loan_grade`, `loan_int_rate`, `loan_status`, etc.) with realistic feature relationships and a ~27.5% default rate. To run on real applicant data, download the [Credit Risk Dataset on Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) and drop it into `data/credit_risk_dataset.csv` — the pipeline code is unchanged.

## Problem
Lenders need to estimate default risk before approving a loan. Missing an actual defaulter (false negative) is more costly than being cautious with a safe applicant (false positive), so the model is optimized with that asymmetry in mind — using `class_weight="balanced"` and evaluating on recall/ROC-AUC rather than raw accuracy.

## Approach
1. **EDA** — distribution analysis, class balance check, correlation matrix, interest rate vs. loan grade
2. **Cleaning** — median imputation for missing employment length; grade-conditional median imputation for missing interest rate
3. **Feature engineering** — income-per-credit-year, loan-to-employment-length ratio, one-hot encoding of categoricals
4. **Modeling** — Logistic Regression, Random Forest, SVM (RBF), combined in a soft-voting ensemble
5. **Evaluation** — confusion matrices, ROC curves, classification reports across all four models
6. **Interpretation** — Random Forest feature importances, plain-language business summary

## Results

| Model | ROC-AUC | Default Recall | Default Precision |
|---|---|---|---|
| Logistic Regression | 0.861 | 0.76 | 0.59 |
| Random Forest | 0.847 | 0.73 | 0.61 |
| SVM (RBF) | 0.840 | 0.76 | 0.59 |
| **Voting Ensemble** | 0.856 | 0.70 | 0.64 |

## Key Insights
- **Loan-to-income ratio and interest rate are the strongest default predictors** — larger income share committed to a loan, and higher-priced loans (which themselves reflect the lender's risk assessment), correlate most with default.
- **Longer credit history is protective**, consistent with standard underwriting logic.
- Class imbalance (~27.5% default rate) means **recall on the Default class matters more than accuracy** — a model that always predicted "no default" would score ~72% accuracy while being useless for risk screening.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## Run it yourself
```bash
pip install -r requirements.txt
jupyter notebook Credit_Risk_Prediction.ipynb
```

## Next Steps
- Threshold tuning based on a lender's actual cost ratio of false negatives vs. false positives
- SHAP values for per-applicant explainability (important for regulated lending)
- Validate findings against the real Kaggle dataset
