import copy
import os
import pickle
import re
import networkx as nx

from caps.components.Cost_estimator.pipeline_estimator import estimate_cost
from caps.components.history_manager import update_and_merge_graphs
from caps.components.parser.parser import add_dataset, split_data, execute_pipeline, extract_artifact_graph
from collections import Counter


def CartesianProduct(sets):
    if len(sets) == 0:
        return [[]]
    else:
        CP = []
        current = sets.popitem()
        for c in current[1]:
            for set in CartesianProduct(sets):
                CP.append(set + [c])
        sets[current[0]] = current[1]
        return CP


def bstar(A, v):
    return A.in_edges(v)


def Expand(A, pi):
    PI = []
    E = {}
    # GET THE EDGES
    for v in [v_prime for v_prime in pi['frontier'] if v_prime not in ['source']]:
        E[v] = bstar(A, v)
    # Find all possible moves
    M = CartesianProduct(E)
    for move in M:
        pi_prime = {
            'cost': pi['cost'],
            'visited': pi['visited'].copy(),
            'frontier': [],
            'plan': pi['plan'].copy()
        }
        for e in move:
            edge_data = A.get_edge_data(*e)
            extra_edges = []
            if 'super' in e[0] or 'split' in e[0]:
                head = list(A.successors(e[0]))
                tail = list(A.predecessors(e[0]))
                extra_edges += list(A.in_edges(e[0]))
                extra_edges += list(A.out_edges(e[0]))
            else:
                head = [e[1]]
                tail = [e[0]]
            # if e[1] not in pi_prime['visited']:
            #    new_nodes = e[1]
            new_nodes = [n for n in head if n not in pi_prime['visited']]
            if new_nodes:
                pi_prime['cost'] += int(10000 * edge_data.get('weight', 0))
                if not extra_edges:
                    pi_prime['plan'].append(e)
                else:
                    pi_prime['plan'] += extra_edges
                pi_prime['visited'].append(new_nodes)
                # if e[0] not in (pi_prime['visited'] + pi_prime['frontier']):
                #    pi_prime['frontier'].append(e[0])
                pi_prime['frontier'] += [n for n in tail if n not in (pi_prime['visited'] + pi_prime['frontier'])]

        PI.append(pi_prime)
    return PI


def exhaustive_optimizer(required_artifacts, history):
    Q = [{'cost': 0, 'visited': [], 'frontier': required_artifacts, 'plan': []}]
    plans = []
    while Q:
        pi = Q.pop(0)
        if pi['frontier'] == ['source']:
            plans.append({'plan': pi['plan'], 'cost': pi['cost']})
        else:
            for pi_prime in Expand(history, pi):
                Q.append(pi_prime)
    return plans


def stack_optimizer(required_artifacts, history):
    Q = [{'cost': 0, 'visited': [], 'frontier': required_artifacts, 'plan': []}]
    cost_star = 99999999999
    pi_star = []
    while Q:
        pi = Q.pop(0)
        # print(pi['frontier'])
        if pi['frontier'] == ['source']:
            if pi['cost'] < cost_star:
                pi_star = pi
                cost_star = pi['cost']
        else:
            plans = Expand(history, pi)
            for pi_prime in plans:
                if pi_prime['cost'] < cost_star:
                    Q.append(pi_prime)
    return pi_star


def extract_steps_from_pipeline(pipeline_str):
    """
    Extract preprocessing steps and the main estimator from the pipeline string using regex.
    Returns a single string representing the combination of all steps in the pipeline.
    """
    # Find all pipeline steps with the format ('step_name', StepType())
    steps = re.findall(r"\('([^']+)',\s*([\w\.]+)\(", pipeline_str)

    # Combine the steps into a single string in the form: "step1_type -> step2_type -> ... -> estimator_type"
    steps_combined = ' -> '.join([step_type for _, step_type in steps])

    return steps_combined


def extract_estimator_from_pipeline(pipeline_str):
    """
    Extract the main estimator from the pipeline string using regex.
    Assumes the pipeline follows the format where the last step's estimator is identifiable.
    """
    estimator_match = re.findall(r"\('.*?',\s*([\w\.]+)\(", pipeline_str)
    estimator = estimator_match[-1] if estimator_match else 'Unknown'
    return estimator


class HistoryGraph:
    def __init__(self, history_id: object, directory: object = None) -> object:
        self.history_id = history_id
        if directory is None:
            directory = 'saved_graphs'
        file_path = os.path.join(directory, f"{self.history_id}.pkl")
        if os.path.exists(file_path):
            # Load the graph if it exists
            with open(file_path, 'rb') as file:
                saved_graph = pickle.load(file)
            self.history = saved_graph.history
            self.eq_history = saved_graph.eq_history
            self.dataset_ids = saved_graph.dataset_ids
            self.evaluated_pipelines = saved_graph.evaluated_pipelines
            self.global_best_error = saved_graph.global_best_error
        else:
            self.global_best_error = float('inf')
            self.history = nx.DiGraph()
            self.eq_history = nx.DiGraph()
            self.history.add_node("source", type="source", size=0, cc=0, alias="storage")
            self.dataset_ids = {}
            self.evaluated_pipelines = Counter()
            self.save_to_file()

    def add_dataset(self, dataset):
        """
               :dataset: A unique identifier for the dataset.
               :param split_ratio: the split ratio to train and test
               """
        self.dataset_ids[dataset] = 0.3

        X, y, self.history, cc = add_dataset(self.history, dataset)
        self.save_to_file()

    # TODO add path to the dataset
    def add_dataset_split(self, dataset, split_ratio):
        """
               :dataset: A unique identifier for the dataset.
               :param split_ratio: the split ratio to train and test
               """
        if dataset not in self.dataset_ids:
            self.dataset_ids[dataset] = split_ratio
            X, y, self.history, cc = add_dataset(self.history, dataset)
            split_data(self.history, dataset, split_ratio, X, y, cc)
            self.save_to_file()

    def save_to_file(self, directory=None):
        """
        Saves the HistoryGraph to a file named after its history_id.
        :param directory: The directory path where the file will be saved. TODO:select directory
        """
        if directory is None:
            directory = 'saved_graphs'  # Default to a 'saved_graphs' subdirectory
            if not os.path.exists(directory):
                os.makedirs(directory)  # Create the directory if it doesn't exist

        file_path = os.path.join(directory, f"{self.history_id}.pkl")
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_from_file(history_id, directory=None):
        """
        Loads a HistoryGraph from a file using its history_id.
        :param history_id: The history_id of the HistoryGraph to be loaded.
        :param directory: The directory path where the file is saved.
        :return: The loaded HistoryGraph object.
        """
        if directory is None:
            directory = 'saved_graphs'

        file_path = os.path.join(directory, f"{history_id}.pkl")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No saved file found for history_id '{history_id}' in '{directory}'")

        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def execute_and_add(self, dataset, pipeline, split_ratio=None):
        if split_ratio == None:
            self.dataset_ids[dataset] = 0.3

        execution_graph, artifacts, request = execute_pipeline(dataset, pipeline, split_ratio)
        self.history = update_and_merge_graphs(copy.deepcopy(self.history), execution_graph)
        # self.history = add_load_tasks_to_the_graph(self.history, artifacts)
        self.save_to_file()
        return request, pipeline

    def delete(self, artifact, mode=None):
        if mode == "all":
            for node, attr in self.history.nodes(data=True):
                if attr.get('type') != 'source' and attr.get('type') != 'training' and attr.get(
                        'type') != 'testing' and attr.get('type') != 'raw' and node != "HIGPPC3810_fit":
                    if self.history.has_edge('source', node):
                        self.history.remove_edge('source', node)

        else:
            if self.history.has_edge('source', artifact):
                self.history.remove_edge('source', artifact)


    def estimate_and_add(self, dataset, pipeline, regression_model="Gradient Boosting", split_ratio=0.3):
        if split_ratio == None:
            self.dataset_ids[dataset] = 0.3
        # X, y = get_dataset(dataset)

        if isinstance(pipeline, str):
            import ast
            from sklearn.pipeline import Pipeline

            cleaned_string = pipeline.replace("\n", " ").replace("  ", " ").strip()
            steps_part = cleaned_string.replace("Pipeline", "").strip("()")
            steps_part = steps_part[len("steps="):] if steps_part.startswith("steps=") else steps_part
            parsed_steps = ast.literal_eval(steps_part)
            pipeline = Pipeline(steps=parsed_steps)

        artifact_graph, request = extract_artifact_graph(dataset, pipeline, None, None)

        train_attributes = self.history.nodes[dataset + "_trainX__"]
        test_attributes = self.history.nodes[dataset + "_testX__"]
        artifact_graph.nodes[dataset + "_trainX__"].update(train_attributes)
        artifact_graph.nodes[dataset + "_testX__"].update(test_attributes)
        pipeline_graph, cost = estimate_cost(artifact_graph, regression_model)
        #print(pipeline)
        #print(cost)
        self.history = update_and_merge_graphs(copy.deepcopy(self.history), pipeline_graph)
        # self.history = add_load_tasks_to_the_graph(self.history, artifacts)
        self.save_to_file()
        return request, pipeline,pipeline_graph, cost