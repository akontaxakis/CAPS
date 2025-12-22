# Function to sample configurations
import os
import random
import re
import time
import matplotlib
import seaborn as sns
import shap
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import ParameterGrid, train_test_split


def error_metrics(test_df: object) -> object:
    # Calculate various error metrics
    test_df['Error'] = test_df['Original'] - test_df['Predicted']
    test_df['Absolute Error'] = test_df['Error'].abs()
    test_df['Squared Error'] = test_df['Error'] ** 2
    test_df['Percentage Error'] = test_df['Error'] / test_df['Original'] * 100
    return test_df


def make_sample(param_grid, n_samples=10):
    sampled_param_grid = random.sample(param_grid, min(n_samples, len(param_grid)))
    return (sampled_param_grid)


def split_configs(param_grid, n_samples=10):
    grid = list(ParameterGrid(param_grid))
    sampled_param_grid = random.sample(grid, min(n_samples, len(grid)))
    remaining_param_grid = [param for param in grid if param not in sampled_param_grid]
    return sampled_param_grid, remaining_param_grid


def all_configs(param_grid):
    grid = list(ParameterGrid(param_grid))
    return grid


# Function to split datasets into training and test datasets
def split_datasets(datasets, position):
    # Select one dataset for testing
    test_dataset = datasets[position]

    # Use the remaining datasets for training
    datasets.pop(position)

    return datasets, test_dataset


def flatten_parameters_column(df, column_name='parameters'):
    flattened_df = df.drop(columns=[column_name]).join(df[column_name].apply(pd.Series))
    return flattened_df


def append_to_csv(df, path):
    if not os.path.isfile(path):
        # File does not exist, so create it
        df.to_csv(path, index=False)
    else:
        # File exists, so append without writing the header
        df.to_csv(path, mode='a', header=False, index=False)


def extracting_training_with_manager(sampling_rate, D, sampled_param_grid, clf_name, clf_class):
    results = []

    # D.data['X_train']
    # D.data['Y_train']
    # D.data['X_valid']
    # D.data['X_test']

    X = D.data['X_train']
    y = D.data['Y_train']

    # Output the shapes of the sampled arrays
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")

    n_features = X.shape[1]
    n_samples = X.shape[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    for param_set in sampled_param_grid:
        clf = clf_class(**param_set)
        start_time = time.time()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        end_time = time.time()
        accuracy = accuracy_score(y_test, y_pred)
        exec_time = end_time - start_time

        results.append({
            'n_features': n_features,
            'n_samples': n_samples,
            'classifier': clf_name,
            'parameters': param_set,
            'accuracy': accuracy,
            'execution_time': exec_time
        })

    return results


def extract_mode(mode):
    # Using regex to extract the required values
    pattern = r'(\w)d(\w)n(\w)f(\w)c'
    match = re.search(pattern, mode)

    if match:
        d, n, f, c = match.groups()
    else:
        print("Pattern not found")
    return d, n, f, c


from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time


# Assuming X and y are already defined
def execute_with_timeout(func, timeout, *args, **kwargs):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            print(f"{func.__name__} timed out after {timeout} seconds.")
            return None


def random_extracting_training_with_manager(
    mode, D, sampled_param_grid, opt_name, opt_class
):
    results = []
    d, n, f, c = extract_mode(mode)
    timeout = 600

    X = D.data['X_train']
    y = D.data['Y_train']

    max_samples = X.shape[0]

    # ---- sample rows ----
    if n == "r":
        n_samples = np.random.randint(1000, max_samples + 1)
        sampled_indices = np.random.choice(X.shape[0], n_samples, replace=False)
        X = X[sampled_indices]
        y = y[sampled_indices]
    else:
        n_samples = max_samples

    # ---- sample features ----
    if f == "r":
        n_features = np.random.randint(1, X.shape[1] + 1)
        feature_indices = np.random.choice(X.shape[1], n_features, replace=False)
        sampled_indices = np.random.choice(X.shape[0], n_samples, replace=False)
        X = X[sampled_indices][:, feature_indices]
        y = y[sampled_indices]
    else:
        n_features = X.shape[1]

    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    for param_set in sampled_param_grid:
        opt = opt_class(**param_set)
        params = opt.get_params()

        # ---------- FIT ----------
        if hasattr(opt, "fit"):
            try:
                start_time = time.time()
                execute_with_timeout(opt.fit, timeout, X_train, y_train)
                fit_time = time.time() - start_time

                results.append({
                    'n_features': X_train.shape[1],
                    'n_samples': X_train.shape[0],
                    'classifier': opt_name,
                    'parameters': params,
                    'execution_time': fit_time,
                    'task': "fit"
                })

            except (ValueError, RuntimeError, TimeoutError):
                continue

        # ---------- TRANSFORM ----------
        if hasattr(opt, "transform"):
            try:
                start_time = time.time()
                execute_with_timeout(opt.transform, timeout, X_train)
                exec_time = time.time() - start_time

                results.append({
                    'n_features': X_train.shape[1],
                    'n_samples': X_train.shape[0],
                    'classifier': opt_name,
                    'parameters': params,
                    'execution_time': exec_time,
                    'task': "transform"
                })

                start_time = time.time()
                execute_with_timeout(opt.transform, timeout, X_test)
                exec_time = time.time() - start_time

                results.append({
                    'n_features': X_test.shape[1],
                    'n_samples': X_test.shape[0],
                    'classifier': opt_name,
                    'parameters': params,
                    'execution_time': exec_time,
                    'task': "transform"
                })

            except (ValueError, RuntimeError, TimeoutError):
                continue

        # ---------- PREDICT ----------
        if hasattr(opt, "predict"):
            try:
                start_time = time.time()
                execute_with_timeout(opt.predict, timeout, X_test)
                exec_time = time.time() - start_time

                results.append({
                    'n_features': X_test.shape[1],
                    'n_samples': X_test.shape[0],
                    'classifier': opt_name,
                    'parameters': params,
                    'execution_time': exec_time,
                    'task': "predict"
                })

            except (ValueError, RuntimeError, TimeoutError):
                continue

    return results


def extracting_training_examples(datasets, sampled_param_grid, clf_name, clf_class):
    results = []
    for dataset in datasets:
        X = dataset.data
        y = dataset.target
        n_features = X.shape[1]
        n_samples = X.shape[0]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        for param_set in sampled_param_grid:
            clf = clf_class(**param_set)
            start_time = time.time()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            end_time = time.time()
            accuracy = accuracy_score(y_test, y_pred)
            exec_time = end_time - start_time

            results.append({
                'n_features': n_features,
                'n_samples': n_samples,
                'classifier': clf_name,
                'parameters': param_set,
                'accuracy': accuracy,
                'execution_time': exec_time
            })
    return results


def plot_predictions(clf_name, model_name, df):
    sns.set_style("whitegrid")
    # Create the plot
    plt.figure(figsize=(12, 8))
    # Plot original and predicted values
    sns.scatterplot(data=df, x=df.index, y='Original', label='Original', marker='o')
    sns.scatterplot(data=df, x=df.index, y='Predicted', label='Predicted', marker='x')
    # Plot the absolute error as bars
    plt.bar(df.index, df['Absolute Error'], color='gray', alpha=0.3, label='Absolute Error')
    # Highlight the errors with error bars
    plt.errorbar(df.index, df['Predicted'], yerr=df['Absolute Error'], fmt='o', color='red', alpha=0.5,
                 label='Error Bars')
    # Add titles and labels
    plt.title(clf_name + ' Original vs Predicted Values(TEST)', fontsize=16)
    plt.xlabel('Index', fontsize=14)
    plt.ylabel('Values', fontsize=14)
    plt.legend()
    # Save the plot to a file

    # Define the path and filename
    path = "experiments/predictions/"
    filename = f"{clf_name}_{model_name}.png"
    # Create the directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    # Save the plot to the specified path
    plt.savefig(os.path.join(path, filename))
    matplotlib.pyplot.close()


def calculate_max_percentage_error(y_true, y_pred):
    """Calculate the maximum percentage error for a single prediction."""
    percentage_errors = ((y_true - y_pred) / y_true) * 100
    max_percentage_error = np.max(percentage_errors)
    return max_percentage_error


def max_error(y_true, y_pred):
    """Calculate the maximum percentage error for a single prediction."""
    percentage_errors = abs(y_true - y_pred)
    max_percentage_error = np.max(percentage_errors)
    return max_percentage_error


def save_error_metrics(model_name, clf_name, y_test, y_test_pred, error_metrics_df, mean_percentage_error_df,
                       max_error_df):
    """Update the error metrics DataFrame."""
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100

    max_erro = max_error(y_test, y_test_pred)

    mae_test = round(mae_test, 3)
    if mape > 999:
        mape = 999
    if mape < 0:
        mape = 0
    max_erro = round(max_erro, 3)

    error_metrics_df.loc[model_name, clf_name] = mae_test
    mean_percentage_error_df.loc[model_name, clf_name] = mape
    max_error_df.loc[model_name, clf_name] = max_erro

    # Round the DataFrames to two decimal places
    return error_metrics_df, mean_percentage_error_df, max_error_df


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_heatmap(error_metrics_df, heatmap_path="error_heatmap.png"):
    """Create and save the heatmap with regression models on the y-axis and classifiers on the x-axis."""
    error_metrics_df = error_metrics_df.apply(pd.to_numeric, errors='coerce')

    def plot_heatmap(df, path, title_suffix=''):
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, annot=True, fmt=".3f", cmap="viridis", cbar_kws={'label': 'Mean Absolute Error'})
        plt.title(f'Heatmap of MAE for Regression Model vs Classifier {title_suffix}')
        plt.xlabel('Classifier')
        plt.ylabel('Regression Model')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    if len(error_metrics_df.columns) > 8:
        half = len(error_metrics_df.columns) // 2
        plot_heatmap(error_metrics_df.iloc[:, :half], heatmap_path.replace(".png", "_part1.png"), ' (Part 1)')
        plot_heatmap(error_metrics_df.iloc[:, half:], heatmap_path.replace(".png", "_part2.png"), ' (Part 2)')
    else:
        plot_heatmap(error_metrics_df, heatmap_path)


def create_mean_percentage_error_heatmap(mean_percentage_error_df, heatmap_path="max_percentage_error_heatmap.png"):
    """Create and save the heatmap for maximum percentage error."""
    mean_percentage_error_df = mean_percentage_error_df.apply(pd.to_numeric, errors='coerce')

    def plot_heatmap(df, path, title_suffix=''):
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=100,
                    cbar_kws={'label': 'Mean Percentage Error (%)'})
        plt.title(f'MAPE for Regression Model vs Operator {title_suffix}')
        plt.xlabel('Operator')
        plt.ylabel('Regression Model')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    if len(mean_percentage_error_df.columns) > 8:
        half = len(mean_percentage_error_df.columns) // 2
        plot_heatmap(mean_percentage_error_df.iloc[:, :half], heatmap_path.replace(".png", "_part1.png"), ' (Part 1)')
        plot_heatmap(mean_percentage_error_df.iloc[:, half:], heatmap_path.replace(".png", "_part2.png"), ' (Part 2)')
    else:
        plot_heatmap(mean_percentage_error_df, heatmap_path)


def create_max_percentage_error_heatmap(max_percentage_error_df, heatmap_path="max_percentage_error_heatmap.png"):
    """Create and save the heatmap for maximum percentage error."""
    max_percentage_error_df = max_percentage_error_df.apply(pd.to_numeric, errors='coerce')

    def plot_heatmap(df, path, title_suffix=''):
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Max Percentage Error (%)'})
        plt.title(f'Heatmap of Max Percentage Error for Regression Model vs Classifier {title_suffix}')
        plt.xlabel('Classifier')
        plt.ylabel('Regression Model')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    if len(max_percentage_error_df.columns) > 8:
        half = len(max_percentage_error_df.columns) // 2
        plot_heatmap(max_percentage_error_df.iloc[:, :half], heatmap_path.replace(".png", "_part1.png"), ' (Part 1)')
        plot_heatmap(max_percentage_error_df.iloc[:, half:], heatmap_path.replace(".png", "_part2.png"), ' (Part 2)')
    else:
        plot_heatmap(max_percentage_error_df, heatmap_path)


def create_max_error_heatmap(max_error_df, heatmap_path="max_error_heatmap.png"):
    """Create and save the heatmap for maximum error."""
    max_error_df = max_error_df.apply(pd.to_numeric, errors='coerce')

    def plot_heatmap(df, path, title_suffix=''):
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Max Error'})
        plt.title(f'Heatmap of Max Error for Regression Model vs Classifier {title_suffix}')
        plt.xlabel('Classifier')
        plt.ylabel('Regression Model')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    if len(max_error_df.columns) > 8:
        half = len(max_error_df.columns) // 2
        plot_heatmap(max_error_df.iloc[:, :half], heatmap_path.replace(".png", "_part1.png"), ' (Part 1)')
        plot_heatmap(max_error_df.iloc[:, half:], heatmap_path.replace(".png", "_part2.png"), ' (Part 2)')
    else:
        plot_heatmap(max_error_df, heatmap_path)


def get_permutation_importances(model, X_test, y_test, model_name, clf_name,
                                output_dir="experiments/feature_importances"):
    """Calculate and save feature importances using permutation importance."""
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    feature_importances = pd.DataFrame(result.importances_mean, index=X_test.columns,
                                       columns=['Importance']).sort_values(by='Importance', ascending=False)

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    feature_importances.to_csv(f"{output_dir}/{clf_name}_{model_name}_permutation_importances.csv")

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances['Importance'], y=feature_importances.index)
    plt.title(f'Permutation Importances for {model_name} with {clf_name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{clf_name}_{model_name}_permutation_importances.png")
    plt.close()


def get_shap_values(model, X_test, model_name, clf_name, output_dir="experiments/feature_importances"):
    """Calculate and save SHAP values for feature importances."""
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)
    shap_importances = pd.DataFrame(shap_values.values, columns=X_test.columns).abs().mean().sort_values(
        ascending=False)

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    shap_importances.to_csv(f"{output_dir}/{clf_name}_{model_name}_shap_importances.csv", header=['Importance'])

    # Plot SHAP values
    plt.figure(figsize=(10, 6))
    sns.barplot(x=shap_importances.values, y=shap_importances.index)
    plt.title(f'SHAP Values for {model_name} with {clf_name}')
    plt.xlabel('SHAP Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{clf_name}_{model_name}_shap_importances.png")
    plt.close()


def random_shape(X, y):
    # Ensure max_samples does not exceed the number of available samples
    max_samples = X.shape[0]
    # Generate a rdrnrfrc number of samples, with a minimum of 1000

    # n_samples = np.random.randint(1000, max_samples + 1)
    n_samples = int(max_samples / 2)
    sampled_indices = np.random.choice(X.shape[0], n_samples, replace=False)
    X = X[sampled_indices]
    y = y[sampled_indices]

    # n_features = np.random.randint(1, X.shape[1] + 1)
    n_features = int(X.shape[1] / 2)
    # Randomly select a set of features
    feature_indices = np.random.choice(X.shape[1], n_features, replace=False)
    # Randomly sample the dataset
    sampled_indices = np.random.choice(X.shape[0], n_samples, replace=False)
    X = X[sampled_indices][:, feature_indices]
    y = y[sampled_indices]

    return X, y


import pandas as pd
import matplotlib.pyplot as plt

def plot_accuracy_graph(dataframe, title="Comparison of TPOT and Hyppo Accuracy Across Datasets"):
    """
    Generate a bar chart comparing TPOT and Hyppo accuracy with benefits labeled.

    Parameters:
        dataframe (pd.DataFrame): Input DataFrame with the following columns:
            - Dataset: Names of the datasets
            - TPOT_P (A): Accuracy values for TPOT
            - Q-C_P (A): Accuracy values for Hyppo
            - Benefit (Hour): Benefits in hours (or 'NR' for not reached)
        title (str): Title of the plot

    Returns:
        None
    """
    x = range(len(dataframe))

    # Bars for TPOT_P and Q-C_P with distinct colors
    fig, ax = plt.subplots(figsize=(10, 6))
    bar1 = ax.bar(x, dataframe["TPOT_P (A)"], width=0.4, label="TPOT Accuracy", align='center', color='orange')
    bar2 = ax.bar([i + 0.4 for i in x], dataframe["Q-C_P (A)"], width=0.4, label="Hyppo Accuracy", align='center', color='skyblue')

    # Add performance values inside the bars
    for bar, val in zip(bar1, dataframe["TPOT_P (A)"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f"{val:.3f}", ha='center', va='center', fontsize=8, color='black')
    for bar, val in zip(bar2, dataframe["Q-C_P (A)"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f"{val:.3f}", ha='center', va='center', fontsize=8, color='black')

    # Add benefit as text above bars
    for i, row in dataframe.iterrows():
        benefit = row["Benefit (Hour)"]
        if benefit == "NR":
            ax.text(i + 0.2, max(row["TPOT_P (A)"], row["Q-C_P (A)"]) + 0.01, "NR", ha='center', fontsize=8, color='red')
        elif pd.notnull(benefit):
            ax.text(i + 0.2, max(row["TPOT_P (A)"], row["Q-C_P (A)"]) + 0.01, f"{benefit:.2f}h", ha='center', fontsize=8)

    # Adjust y-axis scaling for better visibility
    ax.set_ylim(0.35, 1.0)

    # Labels and legend
    ax.set_xticks([i + 0.2 for i in x])
    ax.set_xticklabels(dataframe["Dataset"], rotation=45, ha='right')
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.show()
