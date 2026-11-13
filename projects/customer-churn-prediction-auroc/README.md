# Customer Churn Prediction — AUROC Model Comparison

**Applied machine learning portfolio project | Python · pandas · scikit-learn · supervised classification**

Predicting **which customers are likely to churn** using account, service, and billing information. This project compares five classification algorithms on an identical holdout validation set, examines their ranking quality using **Area Under the Receiver Operating Characteristic Curve (AUROC)**, and produces customer-level probability estimates for an unlabeled Kaggle competition dataset.

The work was completed as graduate coursework in **IDC 6210 (Machine Learning), University of West Florida**. The published notebook contains the original analysis and saved outputs, with presentation notes edited for clarity and an optional local-data loader.

## Result at a glance

**Best observed validation AUROC: 0.8381 — Logistic Regression.**

| Algorithm | Holdout AUROC |
| --- | ---: |
| **Logistic Regression** | **0.8381** |
| Decision Tree | 0.8267 |
| Random Forest | 0.8257 |
| K-Nearest Neighbors | 0.7598 |
| Perceptron | 0.7527 |

The output above is from the notebook's **one stratified 80/20 validation split**, **not** a verified Kaggle leaderboard score. AUROC measures a model's ability to rank a randomly selected positive observation above a negative one; it is not classification accuracy.

### Dataset and outputs

| Measure | Observed |
| --- | ---: |
| Labeled customer records | 5,634 |
| Training / validation customers | 4,507 / 1,127 |
| Overall churn rate | Approximately 27% |
| Original predictive features | 19 |
| Encoded model features | 45 |
| Unlabeled competition customers | 1,409 |
| Churn probability estimates generated | 1,409 |

## What the notebook demonstrates

1. **Separating predictors from labels** — excludes customer identifiers and the churn target from predictive features.
2. **Stratified train/validation split** — keeps churn prevalence similar in both partitions.
3. **Practical preprocessing** — converts text-formatted `total_charges` to numerical values, imputes missing values with the **training-set median**, and one-hot encodes categorical attributes.
4. **Train-only scaling** — fits `StandardScaler` on the encoded training data, then applies the same parameters to validation and unlabeled test data.
5. **Comparable model evaluation** — evaluates Perceptron, Logistic Regression, KNN, Decision Tree, and Random Forest on the same held-out customers.
6. **Correct score inputs for AUROC** — uses `predict_proba()[:, 1]` where supported, or `decision_function()` for Perceptron.
7. **Customer-level predictions** — joins probability estimates to customer IDs **for reporting only** and exports the `id,label` Kaggle submission format.

Linear and distance-based models use scaled features. Decision Tree and Random Forest use the corresponding encoded but unscaled features, which is appropriate because threshold-based trees typically do not require standardization.

## Files

| File | Description |
| --- | --- |
| [customer_churn_prediction.ipynb](customer_churn_prediction.ipynb) | Complete executed Jupyter notebook with explanatory Markdown, five classifiers, validation results, and test predictions |
| [validation_scores.csv](validation_scores.csv) | Measured holdout AUROC scores for the five candidate models |
| [requirements.txt](requirements.txt) | Python package dependencies |
| [.gitignore](.gitignore) | Excludes the original competition datasets, submissions, and local artifacts |

## Reproduce the analysis

The original **Customer Churn Prediction — Fall 2026, IDC 6210** Kaggle competition provides `train.csv` and `test.csv`. These course files **are not redistributed** in this public project.

### In Kaggle

Attach the original competition dataset to the notebook and run the notebook in sequence. The loader automatically recognizes:

```
/kaggle/input/competitions/customer-churn-prediction-fall-2026-idc-6210/
```

The final code cell produces `submission.csv`, containing one ID and one predicted churn probability per unlabeled test observation.

### Locally

1. Obtain `train.csv` and `test.csv` from the course competition.
2. Put them in a `data/` directory alongside the notebook (or set the `CHURN_DATA_DIR` environment variable to their directory).
3. Use Python 3.10+ and install dependencies: `pip install -r requirements.txt`.
4. Run `jupyter notebook customer_churn_prediction.ipynb` and execute cells from top to bottom.

The notebook uses saved Kaggle execution outputs for its displayed results. Actual numerical results may vary across library versions or data changes.

## Model selection and limitations

**Why Logistic Regression?** It attained the strongest AUROC among the five tested baseline configurations on the chosen validation partition. This demonstrates that a comparatively simple regularized linear model can be competitive with ensembles for this dataset.

The analysis is subject to important limitations:

- **Single holdout assessment:** repeated or nested cross-validation was not performed, so the model ranking may depend on the partition.
- **No exhaustive hyperparameter tuning:** initial configurations are compared, not an optimized search.
- **Training subset for test predictions:** the final Logistic Regression model was fitted on **4,507 training rows**, not refitted on all 5,634 labeled observations.
- **No verified Kaggle test performance:** submission probabilities are produced, but a leaderboard score is not provided in the saved notebook.
- **Probability calibration not checked:** the output numbers are model probability estimates, not independently validated real-world risk probabilities.

**Potential next step:** tune with cross-validation using AUROC, evaluate class-level errors and calibration, and retrain the selected model on the full labeled dataset before generating competition predictions.

## Skills demonstrated

`Python` · `pandas` · `NumPy` · `scikit-learn` · `binary classification` · `one-hot encoding` · `median imputation` · `StandardScaler` · `holdout evaluation` · `AUROC` · `Logistic Regression` · `Decision Trees` · `Random Forests` · `KNN` · `reproducible notebooks`

*Educational Kaggle assignment and portfolio demonstration; not a deployed production churn-scoring system.*
