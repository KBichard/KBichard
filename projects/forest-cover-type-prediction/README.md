# Forest Cover Type Prediction — Multiclass Machine Learning

**Portfolio project | Python · scikit-learn · decision trees · random forests · multiclass evaluation**

An annotated graduate machine learning project from the University of West Florida's **IDC 6210** forest-cover Kaggle assignment. The goal is to predict a forest-cover category using geographic and environmental measurements.

## Project objective

This is **supervised multiclass classification**. The labeled training data contain forest-cover class in the `label` column; record `id` is deliberately excluded from model features. An unlabeled test set is used to generate an `id,label` competition submission.

The notebook compares five algorithms on one common stratified **80/20** training/validation split:

| Model | Role |
| --- | --- |
| Perceptron | Linear baseline |
| Logistic Regression | Regularized linear baseline |
| K-Nearest Neighbors (KNN) | Nonlinear distance-based baseline |
| Decision Tree | Interpretable tree splits and interactions |
| Random Forest | Ensemble of randomized trees |

## Methodology

- **Inspect data:** feature types, missing values, duplicate IDs, target-class distribution.
- **Prevent data leakage:** fit median imputation and scaling on the training partition; use fitted transforms for validation.
- **Engineer features:** hydrology distance and hillshade range when source columns exist.
- **Train five classifiers:** linear and KNN methods use scaled continuous features; trees/forest use engineered but unscaled features.
- **Compare models:** training accuracy, validation accuracy, weighted F1, macro F1, and train–validation gap.
- **Interpret errors:** per-class precision, recall, F1, normalized confusion matrix, and feature importance if supported by the selected model.
- **Generate competition predictions:** retrain the chosen estimator on all labeled records and export a checked `submission.csv`.

## Results and reproducibility

The notebook ranks models by **weighted F1**, with validation accuracy as the tie-breaker. When executed it reports the highest-ranked model, the runner-up, the weakest class, and the model's validation accuracy gap.

**Results caveat:** The archived notebook has **no saved execution outputs**. It is not possible to verify its numeric metrics or Kaggle leaderboard performance from this file alone. This README intentionally makes **no claims of a winning model or specific score**. Metrics appear after execution with the original course data. One holdout split is not a substitute for repeated or nested validation.

## Files

```text
forest_cover_type_prediction.ipynb   # Original analysis, with local/Kaggle data-path support
README.md                            # Methods, limitations, and usage
requirements.txt                     # Dependencies
.gitignore                           # Excludes private course data and generated files
```

## Run locally or in Kaggle

1. Obtain the original competition's **`train.csv`** and **`test.csv`** for *Forest Cover Type Prediction — Fall 2026, IDC 6210*. They are **not redistributed** here.
2. Place both files under a **`data/`** folder in your notebook working directory.
3. Install Python 3.10+ and dependencies using `pip install -r requirements.txt`.
4. Open **`forest_cover_type_prediction.ipynb`** in Jupyter or VS Code, then run from the first cell.
5. The final code cell writes **`submission.csv`** with `id,label` columns.

On Kaggle the notebook recognizes its original competition input folder. For a different data location, set `FOREST_COVER_DATA_DIR` to a directory containing both CSV files.

## Limitations and interpretation

- The competition test set is unlabeled, so holdout scores are **not** a leaderboard score.
- The five baseline hyperparameter configurations are not a comprehensive tuning search.
- Relative model performance may change with the split, class frequencies, and evaluation metric.
- Feature importance measures association in the fitted model, not causation.

**Skills:** Python, pandas, NumPy, scikit-learn, preprocessing, multiclass classification, Decision Trees, Random Forests, KNN, Logistic Regression, Perceptron, feature engineering, F1, confusion matrices.

*Graduate coursework portfolio project; not an operational ecological decision-making system.*
