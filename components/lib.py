import inspect
import json
import pickle
import random
import hashlib
import re
import time
import webbrowser

from IPython.display import Image, display
from pyvis.network import Network
from sklearn.decomposition import PCA
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


import pandas as pd
import os
import glob

import os

from caps.components.augmenter import map_node


def list_py_files(directory):
    paths = []
    for root, dirs, files in os.walk(directory):
        for file in glob.glob(os.path.join(root, '*.py')):
            paths.append(file)
    return paths

def load_artifact_graph(artifact_graph, sum, uid, objective, dataset, graph_dir="saved_graphs", mode="eq_"):
    os.makedirs(graph_dir, exist_ok=True)
    file_name = uid + "_AG_" + str(sum) + "_" + mode + objective + "_" + dataset
    ag_path = os.path.join(graph_dir, f"{file_name}.pkl")
    if os.path.exists(ag_path):
        with open(ag_path, 'rb') as f:
            print("load " + ag_path)
            artifact_graph = pickle.load(f)
    return artifact_graph



def store_or_load_artifact_graph(artifact_graph, sum, uid, objective, dataset, graph_dir="saved_graphs"):
    os.makedirs(graph_dir, exist_ok=True)
    file_name = uid + "_AG_" + str(sum) + "_" + objective + "_" + dataset
    ag_path = os.path.join(graph_dir, f"{file_name}.pkl")
    if os.path.exists(ag_path):
        with open(ag_path, 'rb') as f:
            print("load " + ag_path)
            artifact_graph = pickle.load(f)
    else:
        with open(ag_path, 'wb') as f:
            pickle.dump(artifact_graph, f)

    file_name = uid + "_EDGES_AG_" + str(sum) + "_" + objective + "_" + dataset
    ag_path = os.path.join(graph_dir, f"{file_name}.txt")

    with open(ag_path, "w") as outfile:
        # Iterate over edges and write to file
        for u, v, data in artifact_graph.edges(data=True):
            cost = data['cost']
            outfile.write(f'"{u}","{v}",{cost}\n')
    return artifact_graph


def create_artifact_graph(artifacts):
    G = nx.DiGraph()
    for i, (step_name, artifact) in enumerate(artifacts.items()):
        G.add_node(step_name, artifact=artifact)
        if i > 0:
            prev_step_name = list(artifacts.keys())[i - 1]
            G.add_edge(prev_step_name, step_name)
    return G



def load_artifact_graph(artifact_graph, sum, uid, objective, dataset, graph_dir="saved_graphs", mode="eq_"):
    os.makedirs(graph_dir, exist_ok=True)
    file_name = uid + "_AG_" + str(sum) + "_" + mode + objective + "_" + dataset
    ag_path = os.path.join(graph_dir, f"{file_name}.pkl")
    if os.path.exists(ag_path):
        with open(ag_path, 'rb') as f:
            print("load " + ag_path)
            artifact_graph = pickle.load(f)
    return artifact_graph


def extract_artifact_graph(artifact_graph, graph_dir, uid):
    shared_graph_file = uid + "_shared_graph"
    shared_graph_path = os.path.join(graph_dir, f"{shared_graph_file}.plk")
    if os.path.exists(shared_graph_path):
        with open(shared_graph_path, 'rb') as f:
            print("load" + shared_graph_path)
            artifact_graph = pickle.load(f)
    return artifact_graph, shared_graph_path



def keep_two_digits(number):
    str_number = str(number)
    index_of_decimal = str_number.index('.')
    str_number_no_round = str_number[:index_of_decimal + 2]
    return str_number_no_round


def compute_correlation(data1, data2):
    corr_matrix = np.corrcoef(data1, data2, rowvar=False)
    return np.average(np.abs(np.diag(corr_matrix, k=1)))


def compare_pickles_exact(artifact_dir='artifacts'):
    files = [f for f in os.listdir(artifact_dir) if f.endswith('.pkl')]
    num_files = len(files)
    equal_pairs = []

    for i in range(num_files):
        file1 = os.path.join(artifact_dir, files[i])

        with open(file1, 'rb') as f:
            data1 = pickle.load(f)

        for j in range(i + 1, num_files):
            file2 = os.path.join(artifact_dir, files[j])

            with open(file2, 'rb') as f:
                data2 = pickle.load(f)

            if np.array_equal(data1, data2):
                print("found a pair")
                equal_pairs.append((files[i], files[j]))

    return equal_pairs


def compare_pickles(artifact_dir='artifacts', correlation_threshold=0.9):
    files = [f for f in os.listdir(artifact_dir) if f.endswith('.pkl')]
    num_files = len(files)
    highly_correlated_pairs = []
    print(num_files)
    for i in range(num_files):
        file1 = os.path.join(artifact_dir, files[i])

        with open(file1, 'rb') as f:
            data1 = pickle.load(f)

        for j in range(i + 1, num_files):
            file2 = os.path.join(artifact_dir, files[j])

            with open(file2, 'rb') as f:
                data2 = pickle.load(f)

            correlation = compute_correlation(data1, data2)
            print(correlation)
            if correlation >= correlation_threshold:
                highly_correlated_pairs.append((files[i], files[j], correlation))

    return highly_correlated_pairs


def print_metrics(metrics_dir='metrics'):
    n_artifacts = 0;
    os.makedirs(metrics_dir, exist_ok=True)
    file_name = "steps_metrics"
    metrics_path = os.path.join(metrics_dir, f"{file_name}.pkl")
    with open(metrics_path, 'rb') as f:
        print("load " + metrics_path)
        step_times = pickle.load(f)
    for step_name, step_time in step_times:
        if step_name.endswith(("__store", "__score_time")):
            n_artifacts = n_artifacts + 0
        else:
            n_artifacts = n_artifacts + 1
        print("Step '{}' execution time: {}".format(step_name, step_time))
    print("number of artifacts " + str(n_artifacts))


def plot_artifact_graph(G):
    pos = nx.drawing.layout.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10)
    plt.show()


def get_steps(steps):
    mandatory_steps = []
    optional_steps = []
    for step_name, options in steps:
        if (str(step_name)[0].isdigit()):
            optional_steps.append((step_name, options))
        else:
            mandatory_steps.append((step_name, options))
    return optional_steps, mandatory_steps


def get_all_steps(steps):
    mandatory_steps = []
    optional_steps = []
    for step_name, options in steps:
        if (str(step_name)[0].isdigit()):
            optional_steps.append((step_name, options))
        else:
            mandatory_steps.append((step_name, options))
    return optional_steps, mandatory_steps


def get_first_lines(filename, n=10):
    """
    Extract the first n lines of a file.

    Parameters:
    - filename: path to the file
    - n: number of lines to extract

    Returns:
    - list of the first n lines
    """

    with open(filename, 'r', encoding="utf-8") as f:
        lines = [next(f) for _ in range(n)]

    return lines


def fit_pipeline_with_artifacts(pipeline, X_train, y_train):
    artifacts = {}
    X_temp = X_train.copy()

    for step_name, step_transformer in pipeline.steps[:-1]:  # Exclude the classifier step
        X_temp = step_transformer.fit_transform(X_temp, y_train)
        artifacts[step_name] = X_temp.copy()

    # Fit the classifier step
    step_name, step_transformer = pipeline.steps[-1]
    step_transformer.fit(X_temp, y_train)
    artifacts[step_name] = step_transformer

    return artifacts


def create_artifact_graph(artifacts):
    G = nx.DiGraph()

    for i, (step_name, artifact) in enumerate(artifacts.items()):
        G.add_node(step_name, artifact=artifact)
        if i > 0:
            prev_step_name = list(artifacts.keys())[i - 1]
            G.add_edge(prev_step_name, step_name)

    return G


def compute_loading_times(metrics_dir='metrics', artifacts_dir='artifacts'):
    os.makedirs(metrics_dir, exist_ok=True)
    loading_times = {}
    file_name = "loading_metrics"
    metrics_path = os.path.join(metrics_dir, f"{file_name}.pkl")

    if os.path.exists(metrics_path):
        with open(metrics_path, 'rb') as f:
            # print("load " + metrics_path)
            loading_times = pickle.load(f)
    else:
        loading_times = {}

    files = [f for f in os.listdir(artifacts_dir) if f.endswith('.pkl')]

    for file in files:
        file_path = os.path.join(artifacts_dir, file)

        start_time = time.time()
        with open(file_path, 'rb') as f:
            _ = pickle.load(f)
        f.close()
        loading_time = time.time() - start_time
        if (file in loading_times):
            if (loading_time > loading_times[file]):
                loading_times[file] = loading_time
        else:
            loading_times[file] = loading_time
    # print(len(loading_times))
    with open(metrics_path, 'wb') as f:
        pickle.dump(loading_times, f)

    return loading_times


def update_graph(artifact_graph, mem_usage, step_time, param, hs_previous, hs_current, platforms, objective):
    artifact_graph.add_edge(hs_previous, hs_current + "_" + param, type=param, weight=step_time,
                            execution_time=step_time, memory_usage=max(mem_usage), platform=platforms,
                            function=objective)
    return hs_current + "_" + param


def extract_platform(operator):
    split_strings = operator.split('__')
    if (len(split_strings) < 2):
        return "SK"
    else:
        return split_strings[0]


def text_inside_parentheses(s):
    # Find all substrings within parentheses
    matches = re.findall(r'\((.*?)\)', s)
    # Concatenate all matches into a single string, separated by a space (or any other separator you prefer)
    return ' '.join(matches)


def extract_first_two_chars(s, selected_models=[]):
    unified_string = ''.join(selected_models)
    sig = create_4_digit_signature(text_inside_parentheses(s) + unified_string)
    split_strings = s.split('__')
    result = ''.join([substring[:2] for substring in split_strings])
    return result + sig


def create_4_digit_signature(input_string):
    # Create a hash of the input string
    hash_object = hashlib.sha256(input_string.encode())
    hex_dig = hash_object.hexdigest()

    # Convert the hexadecimal hash to an integer
    numeric_hash = int(hex_dig, 16)

    # Reduce the hash to 4 digits. We use modulo 10000 to ensure the result is at most 4 digits
    short_hash = numeric_hash % 10000

    return f"{short_hash:04}"  # Return the number as a zero-padded string


# [plan['cost'],self.history.edge_subgraph(plan['plan']), required_artifacts]
def execute_tasks(tasks_to_execute, memory_artifacts, A, dataset_id):
    # print('memory')
    executed_tasks = []
    # print(list(memory_artifacts.keys()))
    # print('tasks')
    # print(tasks_to_execute)
    trainy = retrieve_artifact(dataset_id + "_trainy__")
    testy = retrieve_artifact(dataset_id + "_testy__")
    #tasks = list(reversed(tasks_to_execute))
    for task in tasks_to_execute:
        data = A.get_edge_data(*task)
        operator = data.get('function', 'No function attribute')
        function = data.get('type', 'No function attribute')
        # print(operator)
        # print(function)
        executed_tasks.append([operator, function])
        node = task[0]
        neighbor = task[1]
        ## FIT
        if function == 'fit':
            args = inspect.signature(operator.fit).parameters
            train_data = memory_artifacts[node]
            requires_y = 'y' in args
            if requires_y:
                fit_result = operator.fit(train_data, trainy)
            else:
                fit_result = operator.fit(train_data)
            memory_artifacts[neighbor] = fit_result

        ## TRANSFORM
        elif 'transform' in function:
            tail = list(A.predecessors(task[0]))
            for item in tail:
                if 'fit' in item:
                    operator = memory_artifacts.get(item)
                else:
                    data_to_transform = memory_artifacts.get(item)
            #print(operator)
            #print(data_to_transform)
            fit_result = operator.transform(data_to_transform)
            memory_artifacts[neighbor] = fit_result

        ## PREDICT
        elif 'predict' in function:
            tail = list(A.predecessors(task[0]))
            for item in tail:
                if 'fit' in item:
                    operator = memory_artifacts.get(item)
                else:
                    test_data = memory_artifacts.get(item)
            predictions = operator.predict(test_data)
            memory_artifacts[neighbor] = predictions
        ## SCORE
        elif 'score' in function:
            fitted_operator = operator.fit(testy)
            predictions = memory_artifacts[node]
            X_temp = fitted_operator.score(predictions)
            memory_artifacts[neighbor] = X_temp
    return memory_artifacts, executed_tasks


def execute_graph(dataset_id, plan):
    A = plan[1].copy()
    required_artifacts = plan[2]

    topo_sort = list(nx.topological_sort(A))
    node_order = {node: i for i, node in enumerate(topo_sort)}

    ### EXECUTING A GRAPH
    pipeline_description = None
    load_tasks = []
    tasks_to_execute = []
    memory_artifacts = {}
    trainy = None
    testy = None

    memory_artifacts[dataset_id + "_trainy__"] = trainy
    memory_artifacts[dataset_id + "_testy__"] = testy
    # load artifacts and add them to memory
    for load_artifacts in A.out_edges('source'):
        memory_artifacts[load_artifacts[1]] = retrieve_artifact(load_artifacts[1])
        load_tasks.append(load_artifacts[1])
    A.remove_node('source')
    queue = []
    visited = []

    for key in memory_artifacts.keys():
        visited.append(key)

    queue += required_artifacts
    Tasks = []
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.append(node)
            tasks = list(A.in_edges(node))
            task = tasks[0]
            extra_edges = []
            if 'super' in task[0] or 'split' in task[0]:
                head = list(A.successors(task[0]))
                tail = list(A.predecessors(task[0]))
                extra_edges += list(A.in_edges(task[0]))
                extra_edges += list(A.out_edges(task[0]))
            else:
                head = [task[1]]
                tail = [task[0]]
            data = A.get_edge_data(*task)
            operator = data.get('function', 'No function attribute')
            function = data.get('type', 'No function attribute')
            tasks_to_execute.append(task)
            for neighbor in tail:
                if neighbor not in visited:
                    queue.append(neighbor)


    sorted_tasks = sorted(tasks_to_execute, key=lambda edge: node_order[edge[0]])

    all_artifacts, executed_tasks = execute_tasks(sorted_tasks, memory_artifacts, A, dataset_id)
    request = all_artifacts[required_artifacts[0]]
    executed_tasks = [["F1ScoreCalculator.score"]]
    return executed_tasks, load_tasks, request


def execute_graph_old(dataset_id, artifact_graph):
    ### EXECUTING A GRAPH
    pipeline_description = None
    memory_artifacts = {}
    train_data = None
    trainy = None
    testy = None
    test_data = None
    trainy = retrieve_artifact(dataset_id + "_trainy__")
    testy = retrieve_artifact(dataset_id + "_testy__")
    memory_artifacts[dataset_id + "_trainy__"] = trainy
    memory_artifacts[dataset_id + "_testy__"] = testy

    if nx.is_directed_acyclic_graph(artifact_graph):
        # Print edges in topological order
        # print("\nEdges in topological order:")
        for node in nx.topological_sort(artifact_graph):
            if artifact_graph.in_degree(node) == 0:
                memory_artifacts[node] = retrieve_artifact(node)
                if 'test' in node:
                    if test_data == None:
                        test_data = memory_artifacts.get(node)
                elif 'train' in node:
                    if train_data == None:
                        train_data = memory_artifacts.get(node)
            for _, neighbor, data in artifact_graph.edges(node, data=True):
                # print(f"({node}, {neighbor})")
                operator = data.get('function', 'No function attribute')
                function = data.get('type', 'No function attribute')
                ## FIT
                if function == 'fit':
                    args = inspect.signature(operator.fit).parameters
                    requires_y = 'y' in args
                    if requires_y:
                        fit_result = operator.fit(train_data, trainy)
                    else:
                        fit_result = operator.fit(train_data)
                    memory_artifacts[neighbor] = fit_result

                ## TRANSFORM
                elif 'transform' in function:
                    last_underscore_index = node.rfind('_')
                    # Slice the string up to the last underscore
                    if last_underscore_index != -1:  # Check if '_' is found
                        operator = memory_artifacts.get(node[:last_underscore_index])
                        if function == 'ftransform':
                            fit_result = operator.transform(train_data)
                            memory_artifacts[neighbor] = fit_result
                            train_data = fit_result
                        elif function == 'tetransform':
                            fit_result = operator.transform(test_data)
                            memory_artifacts[neighbor] = fit_result
                            test_data = fit_result
                ## PREDICT
                elif 'predict' in function:
                    last_underscore_index = node.rfind('_')
                    # Slice the string up to the last underscore
                    if last_underscore_index != -1:  # Check if '_' is found
                        operator = memory_artifacts.get(node[:last_underscore_index])
                    predictions = operator.predict(test_data)
                    memory_artifacts[neighbor] = predictions
                ## SCORE
                elif 'score' in function:
                    fitted_operator = operator.fit(testy)
                    predictions = memory_artifacts[node]
                    X_temp = fitted_operator.score(predictions)
                # print(operator)
                # print(function)
    else:
        print("Graph is not a DAG. Cannot perform topological sort.")


def retrieve_artifact(hs_current, directory=None):
    if directory is None:
        directory = 'artifact_storage'  # Default to a 'artifact_storage' subdirectory
        if not os.path.exists(directory):
            os.makedirs(directory)
    file_path = os.path.join(directory, f"{hs_current}.pkl")
    with open(file_path, 'rb') as file:
        return pickle.load(file)


from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer, MaxAbsScaler, Normalizer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from ConfigSpace import Configuration

def config_to_pipeline(config: Configuration):
    """Train a model based on the configuration provided and return the validation error."""
    # Preprocessor configuration
    seed=0
    preprocessor_name = config['preprocessor']
    if preprocessor_name == 'sklearn.preprocessing.Binarizer':
        preprocessor = Binarizer(threshold=config.get('Binarizer__threshold', 0.0))
    elif preprocessor_name == 'sklearn.preprocessing.MaxAbsScaler':
        preprocessor = MaxAbsScaler()
    elif preprocessor_name == 'sklearn.preprocessing.MinMaxScaler':
        preprocessor = MinMaxScaler()
    elif preprocessor_name == 'sklearn.preprocessing.Normalizer':
        preprocessor = Normalizer(norm=config.get('Normalizer__norm', 'l2'))
    elif preprocessor_name == 'sklearn.preprocessing.StandardScaler':
        preprocessor = StandardScaler()
    elif preprocessor_name == 'sklearn.decomposition.PCA':
        preprocessor = PCA(svd_solver=config.get('PCA__svd_solver', 'randomized'),
                           iterated_power=config.get('PCA__iterated_power', 1))
    else:
        raise ValueError(f"Unknown preprocessor: {preprocessor_name}")

    # Classifier configuration
    classifier_name = config['classifier']
    if classifier_name == 'sklearn.ensemble.RandomForestClassifier':
        model = RandomForestClassifier(
            n_estimators=100,
            max_features=config['RandomForestClassifier__max_features'],
            min_samples_split=config['RandomForestClassifier__min_samples_split'],
            min_samples_leaf=config['RandomForestClassifier__min_samples_leaf'],
            bootstrap=config['RandomForestClassifier__bootstrap'],
            criterion=config['RandomForestClassifier__criterion'],
            random_state=seed
        )
    elif classifier_name == 'sklearn.ensemble.GradientBoostingClassifier':
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=config['GradientBoostingClassifier__learning_rate'],
            max_depth=config['GradientBoostingClassifier__max_depth'],
            min_samples_split=config.get('GradientBoostingClassifier__min_samples_split', 2),
            min_samples_leaf=config.get('GradientBoostingClassifier__min_samples_leaf', 1),
            subsample=config.get('GradientBoostingClassifier__subsample', 1.0),
            max_features=config.get('GradientBoostingClassifier__max_features', None),
            random_state=seed
        )
    elif classifier_name == 'sklearn.ensemble.ExtraTreesClassifier':
        model = ExtraTreesClassifier(
            n_estimators=100,
            max_features=config['ExtraTreesClassifier__max_features'],
            min_samples_split=config['ExtraTreesClassifier__min_samples_split'],
            min_samples_leaf=config['ExtraTreesClassifier__min_samples_leaf'],
            criterion=config['ExtraTreesClassifier__criterion'],
            bootstrap=config['ExtraTreesClassifier__bootstrap'],
            random_state=seed
        )
    elif classifier_name == 'sklearn.tree.DecisionTreeClassifier':
        model = DecisionTreeClassifier(
            criterion=config['DecisionTreeClassifier__criterion'],
            max_depth=config['DecisionTreeClassifier__max_depth'],
            min_samples_split=config.get('DecisionTreeClassifier__min_samples_split', 2),
            min_samples_leaf=config.get('DecisionTreeClassifier__min_samples_leaf', 1),
            random_state=seed
        )
    elif classifier_name == 'sklearn.neighbors.KNeighborsClassifier':
        model = KNeighborsClassifier(
            n_neighbors=config['KNeighborsClassifier__n_neighbors'],
            weights=config['KNeighborsClassifier__weights'],
            p=config['KNeighborsClassifier__p']
        )
    elif classifier_name == 'sklearn.naive_bayes.GaussianNB':
        model = GaussianNB()
    elif classifier_name == 'sklearn.naive_bayes.BernoulliNB':
        model = BernoulliNB(
            alpha=config['BernoulliNB__alpha'],
            fit_prior=config['BernoulliNB__fit_prior']
        )
    elif classifier_name == 'sklearn.naive_bayes.MultinomialNB':
        model = MultinomialNB(
            alpha=config['MultinomialNB__alpha'],
            fit_prior=config['MultinomialNB__fit_prior']
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    # Construct the pipeline with the preprocessor and classifier
    steps = [('preprocessor', preprocessor), ('model', model)]
    pipeline = Pipeline(steps)
    # Return the validation error (1 - mean accuracy)
    return pipeline
