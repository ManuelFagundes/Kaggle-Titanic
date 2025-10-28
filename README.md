# Enhanced Topological Data Analysis (TDA) Pipeline for the Titanic Dataset

A comprehensive data science pipeline for the Titanic survival prediction challenge, distinguishing itself through the integration of **Topological Data Analysis (TDA)** features with advanced classical machine learning techniques.

---

## Overview

This repository contains a complete pipeline, from data loading and cleaning to advanced feature engineering, TDA-based feature extraction, ensemble model training, and final submission generation. The core innovation lies in leveraging TDA (specifically persistent homology) on local neighborhoods of passenger features to derive structural insights, complementing the traditional statistical and domain-specific features.

### Key Components

* **Robust Preprocessing:** Consistent handling of missing values and creation of essential features (`FamilySize`, `IsAlone`, `Title`, etc.).
* **Topological Feature Extraction:** Calculates **Persistent Homology** features (e.g., persistence of $H_0$ components, count of $H_1$ loops) on nearest-neighbor-based point clouds for each passenger.
    * Includes a **statistical fallback** if the `ripser` library is unavailable, ensuring execution stability.
* **Advanced Feature Engineering:** Creation of features like `AgeGroup`, `FarePerPerson`, and `TicketLength`.
* **Robust Ensemble Modeling:** Utilizes a **Soft Voting Classifier** combining **XGBoost**, **Random Forest**, and **LightGBM**, trained with **5-fold Cross-Validation** for stability.

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

## Pipeline Structure

The execution follows a logical sequence, encapsulated within the provided Python cells:

1.  **Dependencies and Setup:** Imports necessary libraries and handles initial warnings. Checks for the availability of `ripser`.
2.  **`load_and_preprocess_titanic`:** Loads the `train.csv` and `test.csv` data, performs basic data cleaning (imputation, category mapping), and normalizes the core feature set.
3.  **`RobustTDAExtractor` Class:**
    * Defines the logic for generating local neighborhoods (using 20 nearest neighbors on scaled features).
    * Computes the $H_0$ (connected components) and $H_1$ (loops) persistence diagrams using `ripser`.
    * Extracts **topological features** (`tda_h0_persistence`, `tda_h1_loops`, etc.) from the diagrams.
    * The **fallback** mechanism uses basic statistical metrics (mean distance, density) if TDA is inaccessible.
4.  **`ConsistentFeatureEngineer` Class:** Performs more elaborate feature creation, ensuring consistent application across both training and test sets.
5.  **`RobustEnsemble` Class:**
    * Handles data scaling (StandardScaler) and cleanup.
    * Trains a **Soft Voting Classifier** using optimized versions of XGBoost, Random Forest, and LightGBM.
    * Selects models based on 5-fold cross-validation accuracy.
6.  **Execution Block:** Coordinates the feature engineering, TDA feature extraction, feature combination (`np.hstack`), ensemble training, prediction, and final submission generation.

---

## Execution and Results

To run the pipeline, ensure the necessary dependencies are installed and the `train.csv` and `test.csv` files are located in the expected directory (`/kaggle/input/titanic/`).

### Output Summary

The final printout provides key performance metrics and data statistics:

* **Cross-Validation Score:** The averaged accuracy across 5 folds for the final ensemble model, indicating the expected out-of-sample performance on the training data.
* **Predicted Survival Rate:** A comparison of the survival rate in the training data to the predicted survival rate in the test data, providing a quick sanity check on class balance.
* **High-confidence predictions:** A metric to assess the model's certainty, counting predictions where the ensemble probability is either below $0.3$ or above $0.7$.

The final predictions are saved to `enhanced_tda_titanic_submission.csv`. This robust, multi-perspective approach attempts to capture not only the explicit feature relationships but also the subtle, topological structure of the data manifold.
