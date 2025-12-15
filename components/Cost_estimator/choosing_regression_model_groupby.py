import os
import logging
import joblib
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, PassiveAggressiveRegressor

# Configure logging
logging.basicConfig(level=logging.INFO)

# Use a non-interactive backend for matplotlib (for headless execution)
matplotlib.use('Agg')

# Set global font style and size for publication-ready figures
size = 36
# Plot settings
plt.rcParams.update({
    'font.size': size,
    'axes.titlesize': size,
    'axes.labelsize': size,
    'xtick.labelsize': size,
    'ytick.labelsize': size,
    'legend.fontsize': size,
    'figure.titlesize': size,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'savefig.dpi': 300,
})

if __name__ == "__main__":
    # Set directories
    training_data_directory = "C:/Users/adoko/PycharmProjects/autoPipe/Cost_estimator/experiments/paper/"
    figures_save_directory = "C:/Users/adoko/PycharmProjects/autoPipe/Cost_estimator/experiments/paper/figures/"
    os.makedirs(figures_save_directory, exist_ok=True)

    # Define tasks
    tasks = ["fit", "transform", "predict"]
    #tasks = ["transform"]

    # Define regression models
    regression_models = {
        "Linear": LinearRegression,
        "Random Forest": RandomForestRegressor,
        "Gradient Boosting": GradientBoostingRegressor,
        "K-Neighbors": KNeighborsRegressor,
        "PassiveAggressive": PassiveAggressiveRegressor,
        "ExtraTrees": ExtraTreesRegressor
    }

    # Operators to analyze
    selected_operators = [
        'sklearn.naive_bayes.GaussianNB',
        'sklearn.naive_bayes.BernoulliNB',
        # 'sklearn.naive_bayes.MultinomialNB',
        'sklearn.tree.DecisionTreeClassifier',
        'sklearn.ensemble.RandomForestClassifier',
        'sklearn.ensemble.GradientBoostingClassifier',
        'sklearn.neighbors.KNeighborsClassifier',
        # 'sklearn.svm.LinearSVC',
        # 'sklearn.linear_model.SGDClassifier',
        'xgboost.XGBClassifier',
        'sklearn.preprocessing.Binarizer',
        'sklearn.decomposition.FastICA',
        # sasssssssssssa'sklearn.cluster.FeatureAgglomeration',
        'sklearn.preprocessing.MaxAbsScaler',
        'sklearn.preprocessing.MinMaxScaler',
        # 'sklearn.preprocessing.Normalizer',
        # 'sklearn.kernel_approximation.Nystroem',
        'sklearn.decomposition.PCA',
        # 'sklearn.preprocessing.PolynomialFeatures',
        'sklearn.kernel_approximation.RBFSampler',
        'sklearn.preprocessing.StandardScaler',
        'tpot.builtins.ZeroCount',
        'tpot.builtins.OneHotEncoder',
        # 'sklearn.feature_selection.SelectFwe',
        # 'sklearn.feature_selection.SelectPercentile',
        'sklearn.ensemble.ExtraTreesClassifier'
        # 'sklearn.feature_selection.VarianceThreshold',
        # 'sklearn.feature_selection.SelectFromModel'

    ]

    # Define time groups
    groups = {
        '<1 sec': lambda y: y < 1,
        '<1 min': lambda y: (y >= 1) & (y < 60),
        '<2 min': lambda y: (y >= 60) & (y < 120),
        '<4 min': lambda y: (y >= 120) & (y < 240),
        '>4 min': lambda y: y >= 240
    }

    # Store best models and predictions
    best_models_predictions = []

    # Step 1: Train models and find best predictions
    for task in tasks:
        for op_name in selected_operators:
            model_training_data = os.path.join(training_data_directory, str(op_name))

            # Load dataset
            try:
                df = pd.read_csv(model_training_data)
            except FileNotFoundError:
                logging.warning(f"File not found: {model_training_data}")
                continue

            df = df[df['task'] == task]
            if df.empty:
                continue

            # Split dataset into train and test sets
            result_train, result_test = [], []
            for _, group in df.groupby('dataset'):
                test_size = 0.1
                group_train, group_test = train_test_split(group, test_size=test_size, random_state=1240)
                result_train.append(group_train)
                result_test.append(group_test)

            if not result_train or not result_test:
                continue

            df_train = pd.concat(result_train, ignore_index=True)
            df_test = pd.concat(result_test, ignore_index=True)

            X_train = pd.get_dummies(df_train.drop(columns=['execution_time', 'dataset', 'classifier', 'task']),
                                     drop_first=True).dropna(axis=1, how='all')
            y_train = df_train['execution_time']
            X_test = pd.get_dummies(df_test.drop(columns=['execution_time', 'dataset', 'classifier', 'task']),
                                    drop_first=True).dropna(axis=1, how='all')
            y_test = df_test['execution_time']

            if y_test.empty or X_train.empty or X_test.empty:
                continue

            best_model, best_model_mape, best_y_test_pred = None, float('inf'), None

            for model_name, ModelClass in regression_models.items():
                model = ModelClass()
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)

                mape_score = mean_absolute_percentage_error(y_test, y_test_pred) * 100

                if mape_score < best_model_mape:
                    best_model_mape = mape_score
                    best_model = model
                    best_y_test_pred = y_test_pred

            if best_model:
                best_models_predictions.append({
                    'Task': task,
                    'Operator': op_name,
                    'y_test': y_test,
                    'y_test_pred': pd.Series(best_y_test_pred, index=y_test.index)  # Ensure Series format
                })

    # Step 2: Compute metrics for each time group
    all_y_test = []
    all_y_test_pred = []

    for entry in best_models_predictions:
        all_y_test.append(entry['y_test'])
        all_y_test_pred.append(entry['y_test_pred'])

    all_y_test = pd.concat(all_y_test, ignore_index=True)
    all_y_test_pred = pd.concat(all_y_test_pred, ignore_index=True)

    # Compute metrics per group
    average_group_metrics = {group: {'MAPE': None, 'MAE': None} for group in groups}

    for group_name, condition in groups.items():
        group_indices = condition(all_y_test)
        if group_indices.any():
            y_test_group = all_y_test[group_indices]
            y_test_pred_group = all_y_test_pred[group_indices]

            y_test_group = y_test_group.replace(0, 0.01)
            y_test_pred_group = y_test_pred_group.replace(0, 0.01)

            average_group_metrics[group_name]['MAPE'] = mean_absolute_percentage_error(y_test_group,
                                                                                       y_test_pred_group) * 100
            average_group_metrics[group_name]['MAE'] = mean_absolute_error(y_test_group, y_test_pred_group)

    # Step 4: Generate publication-quality figures
    group_names = list(groups.keys())
    mape_values = [average_group_metrics[group]['MAPE'] for group in group_names]
    mae_values = [average_group_metrics[group]['MAE'] for group in group_names]

    #colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']


    def save_plot(values, ylabel, filename):
        plt.figure(figsize=(10, 6))
        plt.bar(group_names, values, color=colors, edgecolor='black', linewidth=1.2)
        plt.xlabel('Time Groups')
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_save_directory, filename), format='pdf', bbox_inches='tight')
        plt.close()


    save_plot(mape_values, 'MAPE [%]', 'average_mape_time_groups_v3.pdf')
    save_plot(mae_values, 'MAE [seconds]', 'average_mae_time_groups_v3.pdf')
