# Cost Estimator

This module contains the code used in CAPS to generate, train, and apply predictive models for estimating the execution cost of pipeline components.

Due to GitHub file size limitations, only one dataset (`jannis`) is included directly in the repository. All additional datasets used in the experiments are publicly available at:

https://automl.chalearn.org/data

---

## Overview

CAPS relies on learned cost models to estimate the runtime cost of executing individual pipeline operators (e.g., classifiers, preprocessors, and feature selectors). These estimates are used during pipeline selection to enable cost-aware decision making.

This directory provides:
- code to generate execution-time measurements,
- configuration spaces for supported operators,
- training pipelines for cost prediction models,
- utilities required to integrate these models into CAPS.

---

## Execution-Time Data Generation

The script `extract_train_set.py` generates training data for cost estimation.

For each dataset and each selected operator, the script:

- loads the dataset using the shared `DataManager`,
- randomly samples:
  - subsets of instances,
  - subsets of features,
  - hyperparameter configurations,
- executes the operator for three task types:
  - `fit`,
  - `transform`,
  - `predict`,
- measures wall-clock execution time,
- records execution metadata (number of samples, number of features, operator, parameters),
- stores the results in operator-specific CSV files.

Invalid configurations (e.g., incompatible operator–data combinations or timeouts) are automatically skipped to ensure data collection.

---

## Operator Configuration

The file `config.py` defines:

- `operators_config`: a mapping from operator names to their corresponding scikit-learn (or TPOT/XGBoost) classes,
- `config_params`: hyperparameter configuration spaces used for random sampling.

Meta-estimators such as `SelectFromModel` and `RFE` are handled explicitly using pre-instantiated base estimators to ensure compatibility with scikit-learn’s `ParameterGrid`.

---

## Cost Model Training

The generated execution traces are used to train regression models that predict execution time.

The training procedure:

- loads operator-specific execution data,
- splits data across datasets to avoid information leakage,
- encodes categorical hyperparameters using one-hot encoding,
- trains multiple regression models (e.g., linear models and tree-based ensembles),
- evaluates prediction error using standard metrics,
- serializes trained models for later reuse.

Trained models are stored and reused by CAPS during pipeline selection.

---

## Integration in CAPS

During pipeline search, CAPS queries the trained cost models to estimate the expected cost of executing candidate pipelines before execution. These estimates are combined with performance predictions to guide cost-aware pipeline selection.

All models, configuration spaces, and training code used in the experiments are publicly available and fully reproducible using this module.

---

## Reproducibility Notes

- Only the `jannis` dataset is included due to repository size constraints.
- All other datasets can be downloaded from:
  https://automl.chalearn.org/data
- Random sampling is used during data generation; multiple runs improve robustness.

---



