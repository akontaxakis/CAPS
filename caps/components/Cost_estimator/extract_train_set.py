import os
from matplotlib import pyplot as plt
import pandas as pd

from AutoML_data_manager.data_manager import DataManager
from caps.components.Cost_estimator.config import operators_config, config_params
from caps.components.Cost_estimator.util import split_configs, random_extracting_training_with_manager, append_to_csv, \
    flatten_parameters_column
from pathlib import Path

os.environ["LOKY_MAX_CPU_COUNT"] = "6"
# Set the backend to 'Agg' to avoid issues with GUI backends
plt.switch_backend('Agg')

if __name__ == '__main__':

    model="1d1c"
    PROJECT_ROOT = Path(__file__).resolve().parents[3]  # CAPS/

    datasets_path = PROJECT_ROOT / "datasets"
    training_data_directory = PROJECT_ROOT / "caps" / "components" / "Cost_estimator" / "training_data"
    # Select Operators

    selected_operators = [
        'sklearn.naive_bayes.GaussianNB',
        'sklearn.naive_bayes.BernoulliNB',
        'sklearn.naive_bayes.MultinomialNB',
        'sklearn.tree.DecisionTreeClassifier',
        'sklearn.ensemble.RandomForestClassifier',
        'sklearn.ensemble.GradientBoostingClassifier',
        'sklearn.neighbors.KNeighborsClassifier',
        'sklearn.svm.LinearSVC',
        'sklearn.linear_model.SGDClassifier',
        'xgboost.XGBClassifier',
        'sklearn.preprocessing.Binarizer',
        'sklearn.decomposition.FastICA',
        'sklearn.cluster.FeatureAgglomeration',
        'sklearn.preprocessing.MaxAbsScaler',
        'sklearn.preprocessing.MinMaxScaler',
        'sklearn.preprocessing.Normalizer',
        'sklearn.kernel_approximation.Nystroem',
        'sklearn.decomposition.PCA',
        'sklearn.preprocessing.PolynomialFeatures',
        'sklearn.kernel_approximation.RBFSampler',
        'sklearn.preprocessing.StandardScaler',
        'tpot.builtins.ZeroCount',
        'tpot.builtins.OneHotEncoder',
        'sklearn.feature_selection.SelectFwe',
        'sklearn.feature_selection.SelectPercentile',
        'sklearn.ensemble.ExtraTreesClassifier',
        'sklearn.feature_selection.VarianceThreshold',
        'sklearn.feature_selection.SelectFromModel'
    ]

    datasets = ["jannis"]


    number_of_training_set = 1

    selected_operators = {key: operators_config[key] for key in selected_operators}

    for dataset in datasets:
        D = DataManager(dataset, str(datasets_path), replace_missing=True, verbose=3)

        for clf_name, clf_class in selected_operators.items():
            print(clf_name)
            params = config_params[clf_name]

            for i in range(1, number_of_training_set + 1):
                training_param_grid, _ = split_configs(params, n_samples=1)

                results = random_extracting_training_with_manager(
                    "1d1n1frc", D, training_param_grid, clf_name, clf_class
                )

                # ---- convert to DataFrame safely ----
                if not results:
                    print(f"[SKIP] No results for {clf_name} on {dataset}")
                    continue

                df = pd.DataFrame(results)
                df['dataset'] = dataset

                    # ---- guard against missing execution_time ----
                if 'execution_time' not in df.columns or df['execution_time'].isna().all():
                    print(f"[SKIP] No valid execution_time for {clf_name}")
                    continue

                    # ---- statistics ----
                stats_df = pd.DataFrame([{
                        'min_time': df['execution_time'].min(),
                        'max_time': df['execution_time'].max(),
                        'mean_time': df['execution_time'].mean(),
                        'variance_time': df['execution_time'].var(),
                        'operator': clf_name,
                        'dataset': dataset
                }])

                print(stats_df)

                # ---- save training data ----
                df = flatten_parameters_column(df)
                file_path = os.path.join(training_data_directory, clf_name)
                append_to_csv(df, file_path)

