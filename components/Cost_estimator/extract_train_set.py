import os
from matplotlib import pyplot as plt
import pandas as pd


from caps.components.Cost_estimator.AutoML_data_manager.data_manager import DataManager
from caps.components.Cost_estimator.config import operators_config, config_params
from caps.components.Cost_estimator.util import split_configs, random_extracting_training_with_manager, append_to_csv, \
    flatten_parameters_column

os.environ["LOKY_MAX_CPU_COUNT"] = "6"
# Set the backend to 'Agg' to avoid issues with GUI backends
plt.switch_backend('Agg')

if __name__ == '__main__':

    model="1d1c"
    datasets_path = r"C:\Users\adoko\PycharmProjects\autoPipe\autoML\datasets"
    training_data_directory=r"C:\Users\adoko\PycharmProjects\autoPipe\Cost_estimator\experiments\paper"
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
        'sklearn.decomposition.FastICA'
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
        'sklearn.ensemble.ExtraTreesClassifier'
        'sklearn.feature_selection.VarianceThreshold',
        'sklearn.feature_selection.SelectFromModel'
    ]

    datasets = ["sylvine","philippine","jannis","christine","dilbert","fabert", "albert",  "digits", "dionis"]


    number_of_training_set = 1

    selected_operators = {key: operators_config[key] for key in selected_operators}
    for dataset in datasets:
        D = DataManager(dataset, datasets_path, replace_missing=True, verbose=3)
        for clf_name, clf_class in selected_operators.items():
            print(clf_name)
            j = 0
            while j < 10:
                params = config_params[clf_name]
                # split the operators configuration space
                for i in range(1, number_of_training_set + 1):
                     # execute the classifer for different datasets and configurations
                    # |results| = number_of_datasets * |training_param_grid|
                    training_param_grid, _ = split_configs(params, n_samples=100)  # Sample 100 configurations
                    df = random_extracting_training_with_manager("1d1n1frc", D, training_param_grid, clf_name,clf_class)
                    df = pd.DataFrame(df)
                    df['dataset'] = dataset
                    # Compute statistics
                    min_exec_time = df['execution_time'].min()
                    max_exec_time = df['execution_time'].max()
                    mean_exec_time = df['execution_time'].mean()
                    var_exec_time = df['execution_time'].var()

                    # Print statistics
                    stats = {
                        'min_time': min_exec_time,
                        'max_time': max_exec_time,
                        'mean_time': mean_exec_time,
                        'variance_time': var_exec_time
                    }

                    stats_df = pd.DataFrame([stats])
                    stats_df['operator'] = clf_name
                    stats_df['dataset'] = dataset
                    append_to_csv(stats_df,"experiments/stats")
                    print(stats_df)


                    df = flatten_parameters_column(df)
                    file_path = os.path.join(training_data_directory, str(clf_name))
                    append_to_csv(df, file_path)
                    j=j+1
