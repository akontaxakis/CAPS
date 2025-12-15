import os
import matplotlib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor, \
    ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor, PassiveAggressiveRegressor, \
    RANSACRegressor, HuberRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import joblib
# Set the maximum number of CPUs for parallel processing
from caps.components.Cost_estimator.config import operators_config
from caps.components.Cost_estimator.util import error_metrics, save_error_metrics, create_mean_percentage_error_heatmap

os.environ["LOKY_MAX_CPU_COUNT"] = "6"

def hash_column(df, column_name):
    """Hashes a specified column of a DataFrame into integer values."""
    return df[column_name].apply(lambda x: hash(x) % 10**8)  # Modulo operation to limit the size of the hash


if __name__ == '__main__':
    # Use a non-interactive backend for matplotlib
    matplotlib.use('Agg')
    max_samples = 100000
    training_data_directory = "C:/Users/adoko/PycharmProjects/autoPipe/Cost_estimator/experiments/paper/"
    tasks = ["fit", "transform", "predict"]
    datasets = ["sylvine", "philippine", "jannis", "christine", "dilbert", "fabert", "albert",  "digits", "fabert"]

    # Dictionary of regression models
    regression_models = {
        "Linear": LinearRegression,
        "Random Forest": RandomForestRegressor,
        "Gradient Boosting": GradientBoostingRegressor,
        "K-Neighbors": KNeighborsRegressor,
        "PassiveAggressive": PassiveAggressiveRegressor,
        "ExtraTrees": ExtraTreesRegressor
    }
    # Select classifiers
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
        'sklearn.ensemble.ExtraTreesClassifier'
        'sklearn.feature_selection.VarianceThreshold',
        'sklearn.feature_selection.SelectFromModel'

    ]
    selected_operators = {key: operators_config[key] for key in selected_operators}
    for dataset in datasets:
        error_metrics_df = pd.DataFrame(index=regression_models.keys(), columns=selected_operators.keys())
        mean_percentage_error_df = pd.DataFrame(index=regression_models.keys(), columns=selected_operators.keys())
        max_error_df = pd.DataFrame(index=regression_models.keys(), columns=selected_operators.keys())

        for task in tasks:

            # Split datasets and train models
            for op_name, op_class in selected_operators.items():
                model_training_data = training_data_directory + str(op_name)
                print(model_training_data)
                df = pd.read_csv(model_training_data)
                df = df[df['task'] == task]
                if df.empty:
                    continue
                if df.shape[0] < 1:
                    continue

                df_train =df[df['dataset'] != dataset]
                df_test = df[df['dataset'] == dataset]

                # Split df_test into two halves
                df_test_half_1 = df_test.iloc[:len(df_test) // 2]
                df_test_half_2 = df_test.iloc[len(df_test) // 2:]

                # Add one half of df_test back to df_train
                df_train = pd.concat([df_train, df_test_half_1], ignore_index=True)

                # Keep the other half as df_test
                df_test = df_test_half_2


                X_train = pd.get_dummies(df_train.drop(columns=['execution_time','dataset', 'classifier', 'task']), drop_first=True)
                X_train = X_train.dropna(axis=1, how='all')
                y_train = df_train['execution_time']
                X = pd.get_dummies(df.drop(columns=['execution_time','dataset', 'classifier', 'task']), drop_first=True)
                X = X.dropna(axis=1, how='all')
                y = df['execution_time']
                X_test = pd.get_dummies(df_test.drop(columns=['execution_time','dataset', 'classifier', 'task']),
                                         drop_first=True)
                X_test = X_test.dropna(axis=1, how='all')
                y_test = df_test['execution_time']
                if y_test.shape[0] < 1:
                    continue

                for model_name, ModelClass in regression_models.items():
                    model = ModelClass()
                    model.fit(X_train, y_train)
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)

                    # Create DataFrames for predictions
                    test_df = pd.DataFrame({'Original': y_test, 'Predicted': y_test_pred})
                    train_df = pd.DataFrame({'Original': y_train, 'Predicted': y_train_pred})

                    # Calculate error metrics
                    test_df = error_metrics(test_df)
                    train_df = error_metrics(train_df)

                    # Plot predictions
                    non_zero_indices = y_test != 0
                    filtered_y_test = y_test[non_zero_indices]
                    filtered_y_test_pred = y_test_pred[non_zero_indices]


                    if filtered_y_test.shape[0] < 1:
                        continue

                    # Now call save_error_metrics with the filtered data
                    save_error_metrics(model_name, op_name, filtered_y_test, filtered_y_test_pred, error_metrics_df,
                                           mean_percentage_error_df, max_error_df)

                    # Retrain model on the full dataset
                    model.fit(X, y)
                    joblib.dump(model, f'models/{model_name}_for_{task}_{op_name.replace(".", "_")}.pkl')  # Save the model

                # Create and save the heatmaps
                create_mean_percentage_error_heatmap(mean_percentage_error_df,
                                                    heatmap_path="experiments/paper/estimator/operator_level/"+task+"_"+dataset+"_mape_percentage_error_heatmap.png")
