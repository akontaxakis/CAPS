import time
import networkx as nx
import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_linnerud, load_digits, load_diabetes
from sklearn.model_selection import train_test_split

from AutoML_data_manager.data_manager import DataManager
from caps.components.Cost_estimator.util import random_shape
from caps.components.parser.sub_parser import store_artifact, execute_pipeline_training, pipeline_training, \
    pipeline_evaluation, execute_pipeline_evaluation


def sample(X_train, y_train, rate):
    sample_size = int(X_train.shape[0] * rate)
    # Generate a list of indices based on the length of your training set
    indices = np.arange(X_train.shape[0])
    # Randomly select indices for your sample
    sample_indices = np.random.choice(indices, size=sample_size, replace=False)
    # Use these indices to sample from X_train and y_train
    sample_X_train = X_train[sample_indices]
    sample_y_train = y_train[sample_indices]
    return sample_X_train, sample_y_train

def pipeline_to_graph(dataset_id, pipeline):
    pass

def init_graph(dataset, split_ratio, dataset_multiplier=1):
    # Load the Breast Cancer Wisconsin dataset
    cc = 0
    G = nx.DiGraph()
    G.add_node("source", type="source", size=0, cc=cc)
    start_time = time.time()
    if dataset == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
        X = np.random.rand(100000, 100)
        y = np.random.rand(100000)
        y = (y > 0.5).astype(int)
    elif dataset == "HIGGS":
        data = np.loadtxt('C:/Users/adoko/Downloads/HIGGS.csv', delimiter=',')
        # Extract and modify the first column based on your condition
        # (e.g., setting it to 0 or 1 if it's greater than 0.5)
        y = np.where(data[:, 0] > 0.5, 1, 0).astype(float)

        # Store the original first column in a separate array
        y = data[:, 0].copy()

        # Drop the first column from the data
        X = data[:, 1:]
    elif dataset == "TAXI":
        data = pd.read_csv('C:/Users/adoko/PycharmProjects/pythonProject1/datasets/taxi_train.csv')
        data['trip_duration'] = data['trip_duration'].replace(-1, 0)
        y = data['trip_duration'].values
        X = data.drop('trip_duration', axis=1).values
    else:
        data = pd.read_csv('C:/Users/adoko/Υπολογιστής/BBC.csv')
        data['target'] = data['target'].replace(-1, 0)
        y = data['target'].values
        X = data.drop('target', axis=1).values

    end_time = time.time()
    cc = end_time - start_time
    G.add_node(dataset, type="raw", size=X.size * X.itemsize, cc=cc, frequency=1)
    platforms = ["python"]
    G.add_edge("source", dataset, type="load", weight=end_time - start_time + 0.000001,
               execution_time=end_time - start_time,
               memory_usage=0, platform=platforms)
    return X, y, G, cc


def add_dataset(G, dataset, dataset_multiplier=0.1):
    # Load the Breast Cancer Wisconsin dataset
    cc = 0
    start_time = time.time()
    if dataset == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
    elif dataset == "iris":
        data = load_iris()
        X, y = data.data, data.target
    elif dataset == "diabetes":
        data = load_diabetes()
        X, y = data.data, data.target
    elif dataset == "digits":
        data = load_digits()
        X, y = data.data, data.target
    elif dataset == "linnerud":
        data = load_linnerud()
        X, y = data.data, data.target
    elif dataset == "wine":
        data = load_wine()
        X, y = data.data, data.target
    elif dataset == "HIGGS":
        data = np.loadtxt('C:/Users/adoko/Downloads/HIGGS.csv', delimiter=',', max_rows=100000)
        y = np.where(data[:, 0] > 0.5, 1, 0).astype(float)

        # Store the original first column in a separate array
        y = data[:, 0].copy()

        # Drop the first column from the data
        X = data[:, 1:]
        #print(data.shape)
    elif dataset == "TAXI":
        data = pd.read_csv('C:/Users/adoko/PycharmProjects/pythonProject1/datasets/taxi_train.csv')
        data['trip_duration'] = data['trip_duration'].replace(-1, 0)
        y = data['trip_duration'].values
        X = data.drop('trip_duration', axis=1).values
    elif dataset == "BBC":
        data = pd.read_csv('C:/Users/adoko/Υπολογιστής/BBC.csv')
        data['target'] = data['target'].replace(-1, 0)
        y = data['target'].values
        X = data.drop('target', axis=1).values
    else:
        data = DataManager(dataset, "datasets", replace_missing=True,
                           verbose=3)
        X = data.data['X_train']
        y = data.data['Y_train']
    end_time = time.time()
    cc = end_time - start_time
    G.add_node(dataset, type="raw", size=X.size, cc=cc, frequency=1, alias=dataset)

    platforms = ["python"]
    G.add_edge("source", dataset, type="load", weight=end_time - start_time + 0.000001,
               execution_time=end_time - start_time,
               memory_usage=0, platform=platforms)
    return X, y, G, cc


def get_dataset(dataset):
    # Load the Breast Cancer Wisconsin dataset
    cc = 0
    if dataset == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
    elif dataset == "iris":
        data = load_iris()
        X, y = data.data, data.target
    elif dataset == "diabetes":
        data = load_diabetes()
        X, y = data.data, data.target
    elif dataset == "digits":
        data = load_digits()
        X, y = data.data, data.target
    elif dataset == "linnerud":
        data = load_linnerud()
        X, y = data.data, data.target
    elif dataset == "wine":
        data = load_wine()
        X, y = data.data, data.target
    elif dataset == "HIGGS":
        data = np.loadtxt('C:/Users/adoko/Downloads/HIGGS.csv', delimiter=',', max_rows=10000)
        # Extract and modify the first column based on your condition
        # (e.g., setting it to 0 or 1 if it's greater than 0.5)
        y = np.where(data[:, 0] > 0.5, 1, 0).astype(float)

        # Store the original first column in a separate array
        y = data[:, 0].copy()

        # Drop the first column from the data
        X = data[:, 1:]
        #print(data.shape)
    elif dataset == "TAXI":
        data = pd.read_csv('C:/Users/adoko/PycharmProjects/pythonProject1/datasets/taxi_train.csv')
        data['trip_duration'] = data['trip_duration'].replace(-1, 0)
        y = data['trip_duration'].values
        X = data.drop('trip_duration', axis=1).values
    elif dataset == "BBC":
        data = pd.read_csv('C:/Users/adoko/Υπολογιστής/BBC.csv')
        data['target'] = data['target'].replace(-1, 0)
        y = data['target'].values
        X = data.drop('target', axis=1).values
    else:
        data = DataManager(dataset, "datasets", replace_missing=True,
                           verbose=3)
        X = data.data['X_train']
        y = data.data['Y_train']
    return X, y


def get_split(X, y, split_ratio):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)
    return X_test, X_train, y_test, y_train


def split_data(artifact_graph, dataset, split_ratio, X, y, cc):
    platforms = []
    platforms.append("python")
    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)
    end_time = time.time()
    mem_usage = [0, 0]  # memory_usage(lambda: train_test_split(X, y, test_size=0.2, random_state=42))
    # G.add_node("y_test")
    step_time = end_time - start_time
    cc = cc + step_time

    artifact_graph.add_node(dataset + "_trainX__", type="training", size=X_train.__sizeof__(), cc=cc, frequency=1,
                            alias="trainX", samples=X_train.shape[0], features=X_train.shape[1])

    artifact_graph.add_node(dataset + "_testX__", type="testing", size=X_test.__sizeof__(), cc=cc, frequency=1,
                            alias="testX", samples=X_test.shape[0], features=X_test.shape[1])

    artifact_graph.add_node(dataset + "_trainy__", type="training", size=y_train.__sizeof__(), cc=cc, frequency=1,
                            alias="trainY", samples=y_train.shape[0], features=1)

    artifact_graph.add_node(dataset + "_testy__", type="testing", size=y_test.__sizeof__(), cc=cc, frequency=1,
                            alias="testY", samples=y_test.shape[0], features=1)

    artifact_graph.add_node(dataset + "_split", type="split", size=0, cc=0, frequency=1, alias = "split")

    artifact_graph.add_edge(dataset, dataset + "_split", type="split", weight=step_time, execution_time=step_time,
                            memory_usage=max(mem_usage), platform=platforms)
    artifact_graph.add_edge(dataset + "_split", dataset + "_trainX__", type="split", weight=0.000001, execution_time=0,
                            memory_usage=0, platform=platforms)
    artifact_graph.add_edge(dataset + "_split", dataset + "_testX__", type="split", weight=0.000001, execution_time=0,
                            memory_usage=0, platform=platforms)
    artifact_graph.add_edge(dataset + "_split", dataset + "_trainy__", type="split", weight=0.000001, execution_time=0,
                            memory_usage=0, platform=platforms)
    artifact_graph.add_edge(dataset + "_split", dataset + "_testy__", type="split", weight=0.000001, execution_time=0,
                            memory_usage=0, platform=platforms)
    return artifact_graph, X_test, X_train, y_test, y_train, cc


def execute_pipeline(dataset, pipeline, split_ratio):
    start_time = time.time()
    artifact_graph = nx.DiGraph()
    X, y = get_dataset(dataset)
    X_test, X_train, y_test, y_train = get_split(X, y, split_ratio)
    cc = time.time() - start_time



    artifacts = []

    # STORE ARTIFACT
    artifacts.append(dataset  + "_trainX__")
    store_artifact(dataset  + "_trainX__", X_train)
    artifacts.append(dataset + "_testX__")
    store_artifact(dataset + "_testX__", X_test)
    artifacts.append(dataset + "_trainy__")
    store_artifact(dataset + "_trainy__", y_train)
    artifacts.append(dataset + "_testy__")
    store_artifact(dataset + "_testy__", y_test)

    new_pipeline = clone(pipeline)

    artifact_graph, artifacts, new_pipeline, cc = execute_pipeline_training(artifact_graph, dataset, new_pipeline,
                                                                            artifacts, X_train, y_train, cc)
    artifact_graph, artifacts, request = execute_pipeline_evaluation(artifact_graph, dataset, new_pipeline,
                                                                     artifacts, X_test, y_test, cc)
    return artifact_graph, artifacts, request


def extract_artifact_graph(dataset, pipeline,X=None, y=None):
    artifact_graph = nx.DiGraph()
    artifacts = []
    print(pipeline)
    new_pipeline = clone(pipeline)
    artifact_graph = pipeline_training(artifact_graph, dataset, new_pipeline,X, y)
    artifact_graph, request = pipeline_evaluation(artifact_graph, dataset, new_pipeline)
    return artifact_graph, request


def graph_to_pipeline(artifact_graph):
    pass