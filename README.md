# Enhanced Topological Data Analysis (TDA) Pipeline for the Titanic Dataset (V2.0)

A comprehensive, **corrected** data science pipeline for the Titanic survival prediction challenge. This version ensures the Topological Data Analysis (TDA) component operates on the **full, most informative feature space**, aligning the geometric analysis with the advanced features used for the final ensemble model.

---

## Overview and Logical Correction

This pipeline integrates robust data preparation, sophisticated feature engineering, and a powerful ensemble model. The core principle is leveraging **Topological Data Analysis (TDA)**—specifically persistent homology—on the *local geometry* of passenger features to derive structural insights (e.g., density, clustering, presence of loops/holes in the data manifold).

### Key Correction in V2.0

In the previous version, TDA was inadvertently applied to a simplified set of initial features. This corrected pipeline ensures:

1.  **Advanced Feature Engineering** (`AgeGroup`, `FarePerPerson`, `TicketLength`) is completed first.
2.  The **complete, enriched feature set** is then **scaled**.
3.  **TDA Feature Extraction** is performed on this scaled, complete feature space, maximizing the quality of the derived topological metrics.

### Pipeline Components

* **Initial Data Preprocessing:** Consistent handling of missing values (imputation) and creation of base features (`FamilySize`, `Title`, `HasCabin`).
* **Consistent Feature Engineer:** Creation of advanced, domain-specific features (`AgeGroup`, `FarePerPerson`, `TicketLength`).
* **Robust TDA Extractor:** Calculates Persistent Homology features ($H_0$ and $H_1$) on nearest-neighbor-based point clouds for each passenger using the **full feature set**. A **statistical fallback** is provided if `ripser` is unavailable.
* **Robust Ensemble Modeling:** Utilizes a **Soft Voting Classifier** combining **XGBoost**, **Random Forest**, and **LightGBM**, validated with 5-fold Cross-Validation.

---

## Requirements

The pipeline requires the following dependencies:

| Library | Purpose |
| :--- | :--- |
| `numpy`, `pandas`, `matplotlib` | Core scientific computing and visualization |
| `scikit-learn` | Preprocessing, Scaling, Modeling |
| `xgboost`, `lightgbm` | Gradient Boosting Models for the Ensemble |
| **`ripser`** | **Topological Data Analysis (TDA)** (Highly Recommended) |

---

## Corrected Execution Flow

The corrected sequence is now strictly hierarchical:

1.  **Data Loading:** Initial data is loaded and minimally cleaned (`load_and_preprocess_titanic`).
2.  **Advanced Feature Engineering:** The raw data is passed to `ConsistentFeatureEngineer` to create the full set of features ($\mathbf{X}_{advanced}$).
3.  **Scaling for TDA:** $\mathbf{X}_{advanced}$ is scaled to create $\mathbf{X}_{full\_normalized}$.
4.  **TDA Extraction:** $\mathbf{X}_{full\_normalized}$ is used by `RobustTDAExtractor` to create $\mathbf{X}_{TDA}$.
5.  **Feature Combination:** $\mathbf{X}_{advanced}$ (unscaled) and $\mathbf{X}_{TDA}$ are concatenated to form the final training matrix.
6.  **Model Training:** The `RobustEnsemble` trains the soft voting classifier on the final matrix.

### Output Summary

The final printout provides a measure of expected generalization performance and model confidence:

* **Cross-Validation Score:** Averaged accuracy for the ensemble model (primary performance metric).
* **Predicted Survival Rate:** Sanity check against the training survival rate.
* **High-confidence predictions:** A metric showing the percentage of predictions where the ensemble model was highly certain (probability $<0.3$ or $>0.7$).

The final predictions are saved to `enhanced_tda_titanic_submission.csv`. This corrected workflow ensures all components are utilized optimally, providing a superior foundation for prediction.
