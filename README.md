# Project README: Titanic Survival Prediction using Topological Data Analysis (TDA)

## Overview: $\text{Titanic\ Survival\ Prediction\ (V4.1 \ - \ TDA\ Enhanced)}$

This repository contains a **stabilized machine learning pipeline** developed for the Kaggle Titanic competition. The core innovation of this version (V4.1) is the integration of **Topological Data Analysis (TDA)** via Persistent Homology to generate novel, robust features reflecting the local connectivity and shape of the passenger data manifold. The project emphasizes advanced feature engineering, **memory-efficient processing**, and ensemble modeling.

The objective was to solve the prediction problem from a methodologically distinct viewpoint, leveraging geometric and **topological insights** from the data structure itself.

---

## Requirements

The pipeline requires several standard scientific computing and machine learning libraries, along with the specific TDA library, `ripser`.

| Library | Purpose | Installation Command |
| :--- | :--- | :--- |
| `numpy` | Numerical operations | `pip install numpy` |
| `pandas` | Data manipulation | `pip install pandas` |
| `scikit-learn` | Preprocessing, Scaling, Modeling | `pip install scikit-learn` |
| `xgboost` | Gradient Boosting Model | `pip install xgboost` |
| `lightgbm` | Gradient Boosting Model | `pip install lightgbm` |
| **`ripser`** | **Topological Data Analysis (TDA)** | `pip install ripser` |

---

## Methodology Highlights and Pipeline Structure

The pipeline employs a logical, multi-stage sequence to ensure stability and maximal information extraction:

### **Pipeline Stages**

1.  **Dependencies and Setup:** Imports all necessary libraries, checks for the availability of `ripser`, and establishes the **KaggleHub connection** to define the correct `titanic_path`. (Crucial fix for environment stability).
2.  **`load_and_preprocess_titanic`:** Loads data using the explicit `titanic_path`, performs basic cleaning (imputation, category mapping), and normalizes the core feature set.
3.  **`ConsistentFeatureEngineer` Class:** Performs more elaborate feature creation, including `AgeGroup`, `FarePerPerson`, and `TicketLength`, ensuring consistent application across both training and test sets.
4.  **`RobustTDAExtractor` Class:**
    * Defines logic for generating local neighborhoods on scaled features.
    * Computes the $H_0$ (connected components) and $H_1$ (loops) persistence diagrams using `ripser`.
    * Extracts **topological features** (`tda_h0_persistence`, `tda_h1_loops`, etc.) from the diagrams, with memory management (`gc.collect`) integrated.
    * Includes a **statistical fallback** if the `ripser` library is unavailable, ensuring execution stability.
5.  **Feature Consolidation:** The advanced statistical features are combined with the derived TDA features (`np.hstack`) to create the final, rich feature matrix.
6.  **`RobustEnsemble` Class:**
    * Handles final data scaling (StandardScaler) and cleanup.
    * Trains a **Soft Voting Classifier** combining optimized versions of **XGBoost**, **Random Forest**, and **LightGBM**.
    * Selects models based on 5-fold cross-validation accuracy.

---

## Performance Summary

The model demonstrates strong generalization and stability, validating the complementary value of the TDA features.

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Final Cross-Validation Score** | **0.8339** | Mean accuracy across 5-fold CV on the training data. A stable, high-confidence score. |
| **Predicted Survival Rate (Test)** | **0.378** | Highly aligned with the training set's true survival rate of 0.384, indicating low prediction bias. |
| **High-Confidence Predictions** | **81.8%** | Over four-fifths of the test set predictions were made with a confidence probability greater than $0.7$ or less than $0.3$. |

The resulting submission file, **`enhanced_tda_titanic_submission.csv`**, is the output of this stable ensemble.

---

## Execution

The entire pipeline is contained within the notebook file.

1.  **Credentials:** Ensure your `kaggle.json` file is accessible or that you have successfully authenticated the Kaggle API in your environment.
2.  **Run:** Execute the notebook **sequentially**. The first cell handles the necessary library imports and defines the `titanic_path` via `kagglehub.competition_download('titanic')`, resolving the crucial data dependency.