import argparse

import autosklearn.classification

from AutoML_data_manager.data_manager import DataManager

parser = argparse.ArgumentParser(description="Run SMAC with TPOT configuration.")
parser.add_argument(
    'dataset_id',
    type=str,
    nargs='?',                   # makes it optional
    default='jannis',              # 🔹 default dataset
    help='The dataset ID argument (default: iris)'
)

parser.add_argument(
    'id',
    type=int,
    nargs='?',                   # makes it optional
    default=6666,                   # 🔹 default experiment ID / seed
    help='The experiment ID / random seed argument (default: 0)'
)

# Parse arguments
args = parser.parse_args()
data_id = args.dataset_id
seed = args.id

if __name__ == "__main__":


    # DataManager setup
    datasets_path = r"datasets"

    iris = DataManager(data_id, datasets_path,
                       replace_missing=True,
                       verbose=3)
    X = iris.data['X_train']
    y = iris.data['Y_train']

    include_estimators = [
        "random_forest",
        "gradient_boosting",
        "extra_trees",
        "decision_tree",
        "k_nearest_neighbors",
        "gaussian_nb",
        "bernoulli_nb",
    ]

    include_preprocessors = [
        "binarizer",
        "maxabs_scaler",
        "minmax_scaler",
        "normalizer",
        "standardize",
        "pca",
    ]

    # Initialize Auto-sklearn
     # Initialize Auto-sklearn
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=300000,
        per_run_time_limit=300,
        tmp_folder='./' + data_id + "_autoSKwithACC_" + str(seed),
        disable_evaluator_output=True,
        memory_limit=8192,
        include={
            "classifier": [
                "random_forest",
                "gradient_boosting",
                "extra_trees",
                "decision_tree",
                "k_nearest_neighbors",
                "gaussian_nb",
                "bernoulli_nb",
            ],
            "feature_preprocessor": [
                "pca",
                "no_preprocessing",  # keep the option of not applying a feature preproc
            ],
            "data_preprocessor": [
                "feature_type"  # enables imputation + scaling (standardize, minmax, normalizer, binarizer, maxabs)
            ],
        },
        seed=seed,
        ensemble_size=0,
        initial_configurations_via_metalearning=20,
        delete_tmp_folder_after_terminate=False,
        metric=autosklearn.metrics.accuracy,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 2},
    )

    # Fit the model
    automl.fit(X, y)

