import copy
import json
import math
import os
import pickle
import re

import networkx as nx
import numpy as np
import pandas as pd

from Cost_estimator.pipeline_estimator import estimate_cost
from HYPPO.components.augmenter import new_edges, create_equivalent_graph_without_fit
from HYPPO.components.history_manager import update_and_merge_graphs, add_load_tasks_to_the_graph
from HYPPO.components.lib import pretty_graph_drawing, graphviz_draw, graphviz_simple_draw, view_dictionary
from HYPPO.components.parser.parser import add_dataset, split_data, execute_pipeline, extract_artifact_graph, \
    get_dataset
from HYPPO.components.parser.sub_parser import pipeline_training, pipeline_evaluation
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
        #print(pi['frontier'])
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


def compute_entropy(propa):
    """
    Compute entropy given a list of probabilities.
    """
    entropy = -sum(p * math.log(p) for p in propa if p > 0)
    return entropy


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
            self.flaML_metrics = saved_graph.flaML_metrics
            self.global_best_error = saved_graph.global_best_error
        else:
            self.flaML_metrics = {}
            self.global_best_error = float('inf')
            self.history = nx.DiGraph()
            self.eq_history = nx.DiGraph()
            self.history.add_node("source", type="source", size=0, cc=0, alias="storage")
            self.dataset_ids = {}
            self.evaluated_pipelines = Counter()
            self.save_to_file()

    def update_flaml_metrics(self, pipeline_scores):
        """
        Updates the FlaML metrics (K0, K1, K2, D1, D2) for each learner using the given pipeline scores.

        Parameters:
        - pipeline_scores: List of dictionaries containing 'pipeline', 'score', and 'fitting_time'.
        """
        for record in pipeline_scores:
            pipeline = record['pipeline']
            if isinstance(record['score'], str):
                score = 1
            else:
                score = 1 - record['score']
            fitting_time = record['fitting_time']

            learner = extract_estimator_from_pipeline(str(pipeline))

            if learner not in self.flaML_metrics:
                self.flaML_metrics[learner] = {'K0': 0, 'K1': None, 'K2': None, 'D1': None, 'D2': None}

            # Update K0
            self.flaML_metrics[learner]['K0'] += fitting_time

            # Update D1 and D2
            if self.flaML_metrics[learner]['D1'] is None or score > self.flaML_metrics[learner]['D1']:
                self.flaML_metrics[learner]['D2'] = self.flaML_metrics[learner]['D1']
                self.flaML_metrics[learner]['D1'] = score

                # Update K1 and K2
                self.flaML_metrics[learner]['K2'] = self.flaML_metrics[learner]['K1']
                self.flaML_metrics[learner]['K1'] = self.flaML_metrics[learner]['K0']

            # Update global best error
            self.global_best_error = min(self.global_best_error, score)

        # Save metrics to file
        self.save_to_file()

    def select_pipelines_based_on_diversity(self, pipelines, K):
        selected_pipelines = []
        zero_score_pipelines = []  # To store pipelines with a score of zero
        current_diversity = 0  # Initial diversity is zero
        remaining_pipelines = pipelines.copy()  # Start with all pipelines
        pipelines_selected = 0

        while pipelines_selected < K:
            best_ratio = -1
            best_pipeline = None

            for pipeline in remaining_pipelines:
                # If the score is zero, store the pipeline and continue to the next iteration
                # Extract the estimator from the pipeline
                estimator = extract_estimator_from_pipeline(str(pipeline))
                # Update the estimator counter temporarily to calculate the averages diversity
                temp_counter = self.evaluated_pipelines.copy()
                temp_counter.update([estimator])

                # Calculate probabilities and entropy for the updated set
                total_estimators = sum(temp_counter.values())
                estimator_probabilities = [count / total_estimators for count in temp_counter.values()]
                new_diversity = compute_entropy(estimator_probabilities)

                # Calculate diversity improvement
                diversity_improvement = new_diversity - current_diversity
                # Track the best pipeline based on the highest diversity improvement / cost ratio
                if diversity_improvement > best_ratio:
                    best_ratio = diversity_improvement
                    best_pipeline = pipeline

            # Add the best pipeline to the selected sequence
            if best_pipeline is not None:
                selected_pipelines.append(best_pipeline)
                pipelines_selected += 1

                # Update the current diversity and estimator counter
                selected_estimator = extract_estimator_from_pipeline(str(best_pipeline))
                self.evaluated_pipelines.update([selected_estimator])

                # Update the current diversity
                total_estimators = sum(self.evaluated_pipelines.values())
                estimator_probabilities = [count / total_estimators for count in self.evaluated_pipelines.values()]
                current_diversity = compute_entropy(estimator_probabilities)

                # Remove the selected pipeline from the remaining pipelines
                remaining_pipelines.remove(best_pipeline)
        self.save_to_file()
        return selected_pipelines

    def select_pipelines_based_on_diversity_cost(self, data_id, pipelines, K):
        selected_pipelines = []
        zero_score_pipelines = []  # To store pipelines with a score of zero
        current_diversity = 0  # Initial diversity is zero
        remaining_pipelines = pipelines.copy()  # Start with all pipelines
        pipelines_selected = 0
        pipe_costs = []  # To store the cost of each pipeline

        # Collect the costs for each pipeline
        for pipeline in remaining_pipelines:
            request, pipeline, cost = self.estimate_and_add(data_id, pipeline)
            pipe_costs.append(cost)

        while pipelines_selected < K and remaining_pipelines:
            best_ratio = -1
            best_pipeline = None
            best_idx = 0

            # Iterate over a copy of the list to avoid modifying it during iteration
            for idx, pipeline in enumerate(remaining_pipelines.copy()):
                cost = pipe_costs[idx]
                estimator = extract_estimator_from_pipeline(str(pipeline))

                # Temporarily update the estimator counter to calculate the average diversity
                temp_counter = self.evaluated_pipelines.copy()
                temp_counter.update([estimator])

                # Calculate probabilities and entropy for the updated set
                total_estimators = sum(temp_counter.values())
                estimator_probabilities = [count / total_estimators for count in temp_counter.values()]
                new_diversity = compute_entropy(estimator_probabilities)

                # Calculate diversity improvement
                diversity_improvement = new_diversity - current_diversity

                # Track the best pipeline based on the highest diversity improvement / cost ratio
                if cost == 0:
                    ratio = diversity_improvement / 0.1
                else:
                    ratio = diversity_improvement / cost

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pipeline = pipeline
                    best_idx = idx

            # Add the best pipeline to the selected sequence
            if best_pipeline is not None:
                selected_pipelines.append(best_pipeline)
                pipelines_selected += 1

                # Update the current diversity and estimator counter
                selected_estimator = extract_estimator_from_pipeline(str(best_pipeline))
                self.evaluated_pipelines.update([selected_estimator])

                # Update the current diversity
                total_estimators = sum(self.evaluated_pipelines.values())
                estimator_probabilities = [count / total_estimators for count in self.evaluated_pipelines.values()]
                current_diversity = compute_entropy(estimator_probabilities)

                # Remove the selected pipeline from the remaining pipelines and its associated cost
                remaining_pipelines.pop(best_idx)
                pipe_costs.pop(best_idx)

        self.save_to_file()  # Save any changes made during the selection process
        return selected_pipelines

    def select_pipelines_based_on_performance_diversity_cost(self, data_id, pipelines, K, predecessors_scores):
        selected_indexes = []
        current_diversity = 0  # Initial diversity is zero
        remaining_pipelines = pipelines.copy()  # Start with all pipelines
        pipelines_selected = 0
        pipe_costs = []  # To store the cost of each pipeline

        # Collect the costs for each pipeline
        for pipeline in remaining_pipelines:
            request, pipeline, cost = self.estimate_and_add(data_id, pipeline)
            pipe_costs.append(cost)

        while pipelines_selected < K and remaining_pipelines:
            best_ratio = -1
            best_pipeline = None
            best_idx = 0
            to_remove = 0

            # Iterate over a copy of the list to avoid modifying it during iteration
            for i, pipeline in enumerate(remaining_pipelines.copy()):
                idx = pipelines.index(pipeline)
                cost = pipe_costs[idx]
                score = predecessors_scores[idx]
                estimator = extract_estimator_from_pipeline(str(pipeline))

                # Temporarily update the estimator counter to calculate the average diversity
                temp_counter = self.evaluated_pipelines.copy()
                temp_counter.update([estimator])

                # Calculate probabilities and entropy for the updated set
                total_estimators = sum(temp_counter.values())
                estimator_probabilities = [count / total_estimators for count in temp_counter.values()]
                new_diversity = compute_entropy(estimator_probabilities)

                # Calculate diversity improvement
                diversity_improvement = new_diversity - current_diversity

                # Track the best pipeline based on the highest diversity improvement / cost ratio
                if cost < 1:
                    ratio = (score * diversity_improvement)
                else:
                    ratio = (score * diversity_improvement) / cost

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pipeline = pipeline
                    best_idx = idx
                    to_remove = i
            # Add the best pipeline to the selected sequence
            if best_pipeline is not None:
                selected_indexes.append(best_idx)
                pipelines_selected += 1

                # Update the current diversity and estimator counter
                selected_estimator = extract_estimator_from_pipeline(str(best_pipeline))
                self.evaluated_pipelines.update([selected_estimator])

                # Update the current diversity
                total_estimators = sum(self.evaluated_pipelines.values())
                estimator_probabilities = [count / total_estimators for count in self.evaluated_pipelines.values()]
                current_diversity = compute_entropy(estimator_probabilities)

                # Remove the selected pipeline from the remaining pipelines and its associated cost
                remaining_pipelines.pop(to_remove)

        self.save_to_file()  # Save any changes made during the selection process
        return selected_indexes


    def save_graph_graphml(self):
        for node, data in self.history.nodes(data=True):
            for key, value in data.items():
                if isinstance(value, list):
                    # Convert list to a JSON-formatted string
                    self.history.nodes[node][key] = json.dumps(value)

        # Iterate over all edges in the graph to check and modify the attributes
        for u, v, data in self.history.edges(data=True):
            # Iterate over each attribute in the edge's data dictionary
            for attr_key, attr_value in list(data.items()):
                # Check if the attribute value is of a type that needs to be serialized (e.g., list or dict)
                if attr_key == "function":
                    # Convert the type object to its fully qualified name as a string
                    data[attr_key] = f"k"
                if isinstance(attr_value, (list, dict)):
                    # Convert the value to a JSON string and update the attribute
                    print(data[attr_key])
                    data[attr_key] = json.dumps(attr_value)
                    print(data[attr_key])

        nx.write_graphml(self.history, 'history.graphml')

    def view_datasets(self):
        data = [(key, value) for key, value in self.dataset_ids.items()]

        # Create a DataFrame from the list of tuples
        df = pd.DataFrame(data, columns=['dataset_id', 'split_ratio'])
        # Create a DataFrame
        return df

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

    def visualize(self, type='none', mode='none', load_edges='none', filter_artifact_id=None, filter='None'):
        if 'eq' in filter:
            G = self.eq_history.copy()
        else:
            G = self.history.copy()

        if filter_artifact_id != None:
            if filter_artifact_id in G:
                target_node = filter_artifact_id
                # Get all predecessors and successors of the target node
                predecessors = set(nx.ancestors(G, target_node))
                successors = set(nx.descendants(G, target_node))
                if 'retrieve' in filter:
                    successors = []
                # Create a set that includes the target node, its predecessors, and its successors
                relevant_nodes = predecessors.union(successors).union({target_node})

                # Create a subgraph with these nodes
                G = G.subgraph(relevant_nodes).copy()
                if mode == "simple":
                    graphviz_simple_draw(G)
                else:
                    graphviz_draw(G, type, mode, load_edges)
            else:
                print("dataset id does not exist")
        else:
            if mode == "simple":
                graphviz_simple_draw(G)
            else:
                graphviz_draw(G, type, mode, load_edges, self.history_id)

    def get_dataset_ids(self):
        print(self.dataset_ids)

    def execute_and_add(self, dataset, pipeline, split_ratio=None):
        if split_ratio == None:
            self.dataset_ids[dataset] = 0.3

        execution_graph, artifacts, request = execute_pipeline(dataset, pipeline, split_ratio)
        self.history = update_and_merge_graphs(copy.deepcopy(self.history), execution_graph)
        # self.history = add_load_tasks_to_the_graph(self.history, artifacts)
        self.save_to_file()
        return request, pipeline

    def generate_plans(self, dataset, pipeline, mode='None'):
        artifact_graph = nx.DiGraph()
        artifacts = []
        artifact_graph = pipeline_training(artifact_graph, dataset, pipeline)
        artifact_graph, request = pipeline_evaluation(artifact_graph, dataset, pipeline)
        # print(request)
        required_artifacts, extra_cost_1, new_tasks = new_edges(self.history, artifact_graph)
        # print(required_artifacts)
        if mode == 'with_eq':
            A = self.eq_history.copy()
        else:
            A = self.history.copy()
        A.remove_node(dataset)
        A.remove_node(dataset + "_split")

        # graphviz_draw(A, 'pycharm', 'full')
        plans = exhaustive_optimizer(required_artifacts, A)
        subgraph = []
        i = 0
        for plan in plans:
            subgraph.append([plan['cost'], self.history.edge_subgraph(plan['plan']), required_artifacts])
            # if i>0:
            #    graphviz_draw(self.history.edge_subgraph(plan['plan']), type='pycharm', mode='full')
            # i=i-1
        return subgraph

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

    def get_augmented_graph(self, dataset_id, filter_artifact_id="None", filter="None"):
        if "eq" in filter:
            G = self.eq_history.copy()
        else:
            G = self.history.copy()
        G.remove_node(dataset_id)
        G.remove_node(dataset_id + "_split")
        if filter_artifact_id != None:
            if filter_artifact_id in G:
                target_node = filter_artifact_id
                # Get all predecessors and successors of the target node
                predecessors = set(nx.ancestors(G, target_node))
                successors = set(nx.descendants(G, target_node))
                if 'retrieve' in filter:
                    successors = []
                # Create a set that includes the target node, its predecessors, and its successors
                relevant_nodes = predecessors.union(successors).union({target_node})
                G = G.subgraph(relevant_nodes).copy()
        return G

    def visualize_augmented(self, dataset_id, type='none', mode='none', load_edges='none', filter_artifact_id=None,
                            filter='None'):
        if "eq" in filter:
            G = self.eq_history.copy()
        else:
            G = self.history.copy()
        G.remove_node(dataset_id)
        G.remove_node(dataset_id + "_split")
        if filter_artifact_id != None:
            if filter_artifact_id in G:
                target_node = filter_artifact_id
                # Get all predecessors and successors of the target node
                predecessors = set(nx.ancestors(G, target_node))
                successors = set(nx.descendants(G, target_node))
                if 'retrieve' in filter:
                    successors = []
                # Create a set that includes the target node, its predecessors, and its successors
                relevant_nodes = predecessors.union(successors).union({target_node})

                # Create a subgraph with these nodes
                G = G.subgraph(relevant_nodes).copy()
                if mode == "simple":
                    graphviz_simple_draw(G)
                else:
                    graphviz_draw(G, type, mode, load_edges)
            else:
                print("dataset id does not exist")
        else:
            if mode == "simple":
                graphviz_simple_draw(G)
            else:
                graphviz_draw(G, type, mode, load_edges)

    def retrieve_artifact(self, artifact, mode=None):
        dataset = 'HIGGS'
        if mode == 'with_eq':
            A = self.eq_history.copy()
        else:
            A = self.history.copy()
        A.remove_node(dataset)
        A.remove_node(dataset + "_split")
        if A.has_node(artifact):
            plans = exhaustive_optimizer([artifact], A)
            subgraph = []
            i = 0
            for plan in plans:
                if mode == 'with_eq':
                    subgraph.append([plan['cost'], self.eq_history.edge_subgraph(plan['plan']), [artifact]])
                else:
                    subgraph.append([plan['cost'], self.history.edge_subgraph(plan['plan']), [artifact]])
            return subgraph
        else:
            print('artifact does not exist in history')

    def retrieve_artifacts(self, artifacts, mode=None):
        dataset = 'HIGGS'
        if mode == 'with_eq':
            A = self.eq_history.copy()
        else:
            A = self.history.copy()
        A.remove_node(dataset)
        A.remove_node(dataset + "_split")
        if A.has_node(artifacts[0]):
            plans = exhaustive_optimizer(artifacts, A)
            subgraph = []
            i = 0
            for plan in plans:
                if mode == 'with_eq':
                    subgraph.append([plan['cost'], self.eq_history.edge_subgraph(plan['plan']), artifacts])
                else:
                    subgraph.append([plan['cost'], self.history.edge_subgraph(plan['plan']), artifacts])
            return subgraph
        else:
            print('artifact does not exist in history')

    def optimal_retrieval_plan(self, dataset_id, artifacts, mode=None):
        dataset = dataset_id
        if mode == 'with_eq':
            new_artifacts = []
            for node in artifacts:
                # if artifact_graph.nodes[node]['type'] != "super":
                if ("_fit_" not in node) and ("_fit" not in node) and (
                        "GL" in node or "GP" in node or "TF" in node or "TR" in node or "SK" in node):
                    modified_node = node.replace("GP", "")
                    modified_node = modified_node.replace("TF", "")
                    modified_node = modified_node.replace("TR", "")
                    modified_node = modified_node.replace("SK", "")
                    modified_node = modified_node.replace("GL", "")
                    new_artifacts.append(modified_node)
                else:
                    new_artifacts.append(node)
            artifacts = new_artifacts
            A = self.eq_history.copy()
        else:
            A = self.history.copy()
        # A.remove_node(dataset)
        # A.remove_node(dataset + "_split")
        if A.has_node(artifacts[0]):
            plan = stack_optimizer(artifacts, A)
            if plan == []:
                plan = stack_optimizer(artifacts, A)
            if plan == []:
                plan = stack_optimizer(artifacts, A)
            subgraph = []
            if mode == 'with_eq':
                subgraph.append([plan['cost'], self.eq_history.edge_subgraph(plan['plan']), artifacts])
            else:
                subgraph.append([plan['cost'], self.history.edge_subgraph(plan['plan']), artifacts])
            return subgraph
        else:
            print('artifact does not exist in history')

    def find_equivalent_operators(self):
        self.eq_history = create_equivalent_graph_without_fit(self.history)

    def sort_by_metrics(self, dataset_id, metric):
        G = self.history.copy()
        value_set = set()
        if dataset_id in G:
            target_node = dataset_id
            # Get all predecessors and successors of the target node
            predecessors = set(nx.ancestors(G, target_node))
            successors = set(nx.descendants(G, target_node))
            # Create a set that includes the target node, its predecessors, and its successors
            relevant_nodes = predecessors.union(successors).union({target_node})

            # Create a subgraph with these nodes
            G = G.subgraph(relevant_nodes).copy()
            # Iterate through the nodes to process those with type 'score'
            for node, attr in G.nodes(data=True):
                if attr.get('type') == 'score':
                    for node, attr in G.nodes(data=True):
                        if attr.get('type') == 'score' and attr.get('operator') == metric:
                            operator = attr.get('operator')
                            value = attr.get('value', 0)  # Default value is 0 if not present
                            # Update the highest and lowest value for each operator
                            value_set.add(value)
        return sorted(list(value_set), reverse=True)

    def best_metrics_achieved(self, dataset_id):
        G = self.history.copy()
        if dataset_id in G:
            target_node = dataset_id
            # Get all predecessors and successors of the target node
            predecessors = set(nx.ancestors(G, target_node))
            successors = set(nx.descendants(G, target_node))
            # Create a set that includes the target node, its predecessors, and its successors
            relevant_nodes = predecessors.union(successors).union({target_node})

            # Create a subgraph with these nodes
            G = G.subgraph(relevant_nodes).copy()
            from collections import defaultdict
            highest_values = defaultdict(lambda: float('-inf'))
            lowest_values = defaultdict(lambda: float('inf'))

            # Iterate through the nodes to process those with type 'score'
            for node, attr in G.nodes(data=True):
                if attr.get('type') == 'score':
                    for node, attr in G.nodes(data=True):
                        if attr.get('type') == 'score':
                            operator = attr.get('operator')
                            value = attr.get('value', 0)  # Default value is 0 if not present
                            # Update the highest and lowest value for each operator
                            highest_values[operator] = max(highest_values[operator], value)
                            lowest_values[operator] = min(lowest_values[operator], value)

        import pandas as pd
        df = pd.DataFrame([(operator, highest, lowest) for operator, highest, lowest in
                           zip(highest_values, highest_values.values(), lowest_values.values())],
                          columns=['operator', 'highest_value', 'lowest_value'])
        return df

    def compareHistoryAgainstDictionary(self, dataset_id, objective):

        dictionary = view_dictionary(objective=objective)
        popular_operators = self.popular_operators(dataset_id)

        num_rows = dictionary.shape[0]

        # Generate random integer values for each row
        # For example, random integers between 1 and 100
        dictionary['Frequency'] = np.random.randint(10, 101, size=num_rows)

        # For df2, remove details within parentheses to match 'Implementation' format
        popular_operators['NormalizedOperator'] = popular_operators['Operator'].str.extract(r'([^\(]+)')

        # Now, filter df1 by implementations that do not exist in the normalized 'Operator' of df2
        # We're using ~df1['Implementation'].isin(df2['NormalizedOperator']) to find non-matching rows
        df3 = dictionary[~dictionary['Implementation'].isin(popular_operators['NormalizedOperator'])][
            ['Implementation', 'Frequency']]
        return df3

    def retrieve_best_pipelines(self, dataset_id, metric, N):
        A = self.history.copy()
        df = pd.DataFrame(columns=[metric, "Pipeline"])  # Main DataFrame

        highest_values = self.sort_by_metrics(dataset_id, metric)

        edges_to_remove = [(u, v) for u, v in A.out_edges("source") if
                           A.nodes[v].get('type') not in ['training', 'testing']]
        A.remove_edges_from(edges_to_remove)
        for i in range(min(N, len(highest_values))):  # Ensure loop does not exceed available highest values
            G = A.copy()
            specific_value = highest_values[i]  # No need to check for None, loop controls it

            request = None
            for node, attr in G.nodes(data=True):
                if attr.get('operator') == metric and attr.get('value') == specific_value:
                    request = node
                    break

            if request is None:
                print(f"No node found with operator '{metric}' and value '{specific_value}'")
                continue

            G.remove_node(dataset_id)
            G.remove_node(dataset_id + "_split")

            # Assuming exhaustive_optimizer and its output handling are correct
            plans = exhaustive_optimizer([request], G)
            if not plans:
                continue
            plan = plans.pop(0)
            graph = self.history.edge_subgraph(plan['plan'])

            # Extract the pipeline process based on your specific logic
            topological_sorted_nodes = list(nx.topological_sort(graph))
            filtered_sorted_aliases = [graph.nodes[node]['alias'] for node in topological_sorted_nodes
                                       if 'type' in graph.nodes[node] and graph.nodes[node]['type'] != "super"
                                       and 'alias' in graph.nodes[node]
                                       and graph.nodes[node]['alias'] not in {"predictions", "trainX", "storage",
                                                                              "testX"}]

            unique_array = []
            for item in filtered_sorted_aliases:
                if item not in unique_array:
                    unique_array.append(item)

            # Create a averages DataFrame for the current data point
            new_data = {metric: [specific_value], "Pipeline": [unique_array]}
            new_df = pd.DataFrame(new_data)

            # Concatenate the new_df to the main df
            df = pd.concat([df, new_df], ignore_index=True)

        return df

    def retrieve_best_pipeline(self, dataset_id, metric, mode='review'):
        G = self.history.copy()
        metrics = self.best_metrics_achieved(dataset_id)
        highest_value_for_specific_operator = metrics[metrics['operator'] == metric]['highest_value'].max()

        specific_operator = metric
        specific_value = highest_value_for_specific_operator
        request = None
        for node, attr in G.nodes(data=True):
            if attr.get('operator') == specific_operator and attr.get('value') == specific_value:
                request = node
                break
        if request == None:
            print(f"No node found with operator '{specific_operator}' and value '{specific_value}'")

        G.remove_node(dataset_id)
        G.remove_node(dataset_id + "_split")
        # graphviz_draw(G, 'pycharm', 'full')
        if (mode == 'review'):
            edges_to_remove = [(u, v) for u, v in G.out_edges("source") if
                               G.nodes[v].get('type') not in ['training', 'testing']]

        # Remove the identified edges
        G.remove_edges_from(edges_to_remove)

        plans = exhaustive_optimizer([node], G)
        plan = plans.pop(0)
        graph = self.history.edge_subgraph(plan['plan'])

        # Perform a topological sort on the graph
        topological_sorted_nodes = list(nx.topological_sort(graph))

        # Extracting the 'alias' attribute of each node, following the topological order
        # Exclude nodes with specific types or aliases
        filtered_sorted_aliases = [graph.nodes[node]['alias'] for node in topological_sorted_nodes
                                   if 'type' in graph.nodes[node] and graph.nodes[node]['type'] != "super"
                                   and 'alias' in graph.nodes[node]
                                   and graph.nodes[node]['alias'] not in {"predictions", "trainX", "storage", "testX"}]

        unique_array = []
        for item in filtered_sorted_aliases:
            if item not in unique_array:
                unique_array.append(item)
        return unique_array

    def retrieve_pipelines(self, dataset_id, score_operator="AccuracyCalculator", threshold=0.6, mode="review"):
        G = self.history.copy()
        if dataset_id in G:
            target_node = dataset_id
            # Get all predecessors and successors of the target node
            predecessors = set(nx.ancestors(G, target_node))
            successors = set(nx.descendants(G, target_node))
            # Create a set that includes the target node, its predecessors, and its successors
            relevant_nodes = predecessors.union(successors).union({target_node})

            # Create a subgraph with these nodes
            G = G.subgraph(relevant_nodes).copy()
            from collections import defaultdict
            highest_values = defaultdict(lambda: float('-inf'))
            nodes = []
            # Iterate through the nodes to process those with type 'score'
            for node, attr in G.nodes(data=True):
                if attr.get('type') == 'score':
                    operator = attr.get('operator')
                    value = attr.get('value', 0)
                    if operator == score_operator and value >= threshold:
                        nodes.append(node)

        #print(nodes)
        G.add_node("request", type="super", size=0, cc=0)
        for node in nodes:
            G.add_edge(node, 'request', type='super', weight=0,
                       execution_time=0, memory_usage=0, platform=["None"],
                       function=None)

        G.remove_node(dataset_id)
        G.remove_node(dataset_id + "_split")

        if (mode == 'review'):
            edges_to_remove = [(u, v) for u, v in G.out_edges("source") if
                               G.nodes[v].get('type') not in ['training', 'testing']]

            # Remove the identified edges
        G.remove_edges_from(edges_to_remove)

        plans = exhaustive_optimizer(["request"], G)
        subgraph = []
        i = 0
        for plan in plans:
            subgraph.append([plan['cost'], self.history.edge_subgraph(plan['plan']), ["request"]])
        return subgraph

    def retrieve_optimal_pipelines(self, dataset_id, score_operator="AccuracyCalculator", threshold=0.6, mode="review"):
        G = self.history.copy()
        if dataset_id in G:
            target_node = dataset_id
            # Get all predecessors and successors of the target node
            predecessors = set(nx.ancestors(G, target_node))
            successors = set(nx.descendants(G, target_node))
            # Create a set that includes the target node, its predecessors, and its successors
            relevant_nodes = predecessors.union(successors).union({target_node})

            # Create a subgraph with these nodes
            G = G.subgraph(relevant_nodes).copy()
            from collections import defaultdict
            highest_values = defaultdict(lambda: float('-inf'))
            nodes = []
            # Iterate through the nodes to process those with type 'score'
            for node, attr in G.nodes(data=True):
                if attr.get('type') == 'score':
                    operator = attr.get('operator')
                    value = attr.get('value', 0)
                    if operator == score_operator and value >= threshold:
                        nodes.append(node)

        G.add_node("request", type="super", size=0, cc=0)
        for node in nodes:
            G.add_edge(node, 'request', type='super', weight=0,
                       execution_time=0, memory_usage=0, platform=["None"],
                       function=None)

        G.remove_node(dataset_id)
        G.remove_node(dataset_id + "_split")

        if (mode == 'review'):
            edges_to_remove = [(u, v) for u, v in G.out_edges("source") if
                               G.nodes[v].get('type') not in ['training', 'testing']]

            # Remove the identified edges
        G.remove_edges_from(edges_to_remove)

        plan = stack_optimizer(["request"], G)
        subgraph = []
        subgraph.append([plan['cost'], self.history.edge_subgraph(plan['plan']), ["request"]])
        return subgraph

    def equivalent_operators(self, dataset_id, pair):
        self.find_equivalent_operators()

    def printArtifacts(self, mode=None):
        if mode == None:
            for node in self.history.nodes:
                print(node)
        elif mode == "with_eq":
            for node in self.eq_history.nodes:
                print(node)

    def popular_operators(self, dataset_id):
        G = self.get_augmented_graph(dataset_id, dataset_id)

        # Step 1: Extract Node Frequencies
        node_frequencies = {node: G.nodes[node]['frequency'] for node in G.nodes if node != "source"}
        node_alias = {node: G.nodes[node]['alias'] for node in G.nodes if node != "source"}
        # Step 2: Calculate Operator Frequencies
        operator_frequency = {}
        for edge in G.edges:
            source_node, target_node = edge
            # Skip edges that originate from "source"
            if source_node == "source":
                continue
            if G.edges[edge]['type'] == "super":
                continue
            target_node = edge[1]
            operator = node_alias[target_node]
            # Assuming the edge is directed and you want the frequency of the node the edge points to
            if operator == "predictions":
                continue
            frequency = node_frequencies[target_node]
            if operator in operator_frequency:
                operator_frequency[operator] += frequency
            else:
                operator_frequency[operator] = frequency

        # Step 3: Rank Operators
        # Sort the operators based on the summed frequency
        operators_ranked = sorted(operator_frequency.items(), key=lambda x: x[1], reverse=True)
        df_operators_ranked = pd.DataFrame(operators_ranked, columns=['Operator', 'Frequency'])
        df_filtered = df_operators_ranked[
            ~(df_operators_ranked['Operator'].isin(['trainX', 'testX', 'split', 'trainY', 'testY']))
        ]
        return df_filtered

    def optimize_pipeline(self, dataset_id, pipeline):
        graph, request = extract_artifact_graph(dataset_id, pipeline)
        plans = self.optimal_retrieval_plan(dataset_id, [request])
        return plans.pop(0)

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
        return request, pipeline, cost

    def prune(self, K, pipelines_summary):
        execution_graph = nx.DiGraph()
        temp_history = copy.deepcopy(self.history)
        pipelines_summary_true = pd.DataFrame(columns=['Pipe', 'Request', 'Chosen'])
        choosen = 0
        total_cost = 0
        choosen_request = []
        while choosen < K and choosen < len(pipelines_summary):
            minimum_path, cost, c_r = self.choose_minimum(pipelines_summary)
            total_cost = total_cost + cost
            choosen_request = choosen_request + c_r
            for index, row in pipelines_summary.iterrows():
                if row['Request'] in choosen_request:
                    pipelines_summary.at[index, 'Chosen'] = 1
            execution_graph = update_and_merge_graphs(execution_graph, minimum_path)
            choosen = choosen + 1
        return execution_graph, pipelines_summary, total_cost

    def choose_minimum(temp_history, pipelines_summary):
        requests_not_choosen_yet = pipelines_summary[pipelines_summary['Chosen'] == 0]['Request']
        minimum_cost = 10000000
        minimum_path = None
        minimum_request = []
        for request in requests_not_choosen_yet:
            plans = temp_history.optimal_retrieval_plan("jannis", [request], mode=None)
            plan = plans.pop(0)
            cost = plan[0]
            required_artifacts = plan[2]
            if cost < minimum_cost:
                minimum_cost = cost
                minimum_path = plan[1].copy()
                minimum_request = plan[2]

            # print(request)

        return minimum_path, minimum_cost, minimum_request

    import pandas as pd
    import numpy as np

    def select_pipelines_pareto_then_performance(self, pipelines_summary, sklearn_pipeline_list, K):
        selected_pipelines = []
        remaining_pipelines = pipelines_summary.copy()
        selected_count = 0  # Keep track of how many pipelines have been selected

        while selected_count < K:
            # Step 1: Initialize a boolean array using the index of the DataFrame
            is_pareto_optimal = pd.Series(True, index=remaining_pipelines.index)

            # Step 2: Iterate over all pipelines and compare them to find the Pareto front
            for i, pipeline_i in remaining_pipelines.iterrows():
                for j, pipeline_j in remaining_pipelines.iterrows():
                    if (pipeline_j['Performance'] > pipeline_i['Performance']) and (
                            pipeline_j['Cost'] < pipeline_i['Cost']):
                        # Pipeline_j dominates pipeline_i, so mark pipeline_i as not Pareto optimal
                        is_pareto_optimal[i] = False
                        break

            # Step 3: Get the Pareto front pipelines
            pareto_pipelines_df = remaining_pipelines[is_pareto_optimal]

            # Step 4: Add the Pareto front pipelines to the selected pipelines list
            selected_pipelines.append(pareto_pipelines_df)
            selected_count += len(pareto_pipelines_df)  # Update the count of selected pipelines

            # Step 5: Remove the Pareto front pipelines from the remaining set
            remaining_pipelines = remaining_pipelines[~is_pareto_optimal]

            # Step 6: Check if we have selected more than or exactly K pipelines
            if selected_count >= K:
                break

        # Step 7: Concatenate the selected Pareto fronts
        selected_pipelines_df = pd.concat(selected_pipelines)

        # Step 8: If we have more than K, sort by performance and select the top K
       # if len(selected_pipelines_df) > K:
        #    selected_pipelines_df = selected_pipelines_df.sort_values(by='Performance', ascending=False).head(K)

        # Step 9: Find the indexes of the selected pipelines in the original sklearn_pipeline_list
        selected_indexes = [sklearn_pipeline_list.index(pipe) for pipe in selected_pipelines_df['Pipe']]

        return selected_indexes


    def select_pipelines_pareto_then_performance_agressive(self, pipelines_summary, sklearn_pipeline_list):
        # Initialize an array to mark Pareto-optimal pipelines
        is_pareto_optimal = pd.Series(True, index=pipelines_summary.index)

        # Iterate over all pipelines and compare them to find the Pareto front
        for i, pipeline_i in pipelines_summary.iterrows():
            for j, pipeline_j in pipelines_summary.iterrows():
                if (pipeline_j['Performance'] > pipeline_i['Performance']) and (
                        pipeline_j['Cost'] < pipeline_i['Cost']):
                    # Pipeline_j dominates pipeline_i, so mark pipeline_i as not Pareto optimal
                    is_pareto_optimal[i] = False
                    break

        # Get the Pareto front pipelines
        pareto_pipelines_df = pipelines_summary[is_pareto_optimal]

        # Find the indexes of the Pareto-optimal pipelines in the original sklearn_pipeline_list
        selected_indexes = [sklearn_pipeline_list.index(pipe) for pipe in pareto_pipelines_df['Pipe']]

        return selected_indexes


    def AutoMLselection(self,data_id, N, K, sklearn_pipeline_list, l, predecessor_scores, timeout=None):
        import pandas as pd
        if N == 2:

            pipelines_summary = pd.DataFrame(columns=['Pipe', 'Cost'])

            for idx, sklearn_pipeline in enumerate(sklearn_pipeline_list):
                request, pipeline, cost = self.estimate_and_add(data_id, sklearn_pipeline)
                # X, y = get_dataset(data_id)
                # artifact_graph, request = extract_artifact_graph(data_id, sklearn_pipeline, features, target)
                graph_summary = pd.DataFrame({'Pipe': [sklearn_pipeline], 'Cost': [cost]})
                pipelines_summary = pd.concat([pipelines_summary, graph_summary], ignore_index=True)

            sorted_pipelines = pipelines_summary.sort_values(by=['Cost'], ascending=[True])
            selected_pipelines_df = sorted_pipelines.head(K)
            selected_indices_set = [sklearn_pipeline_list.index(pipe) for pipe in selected_pipelines_df['Pipe']]
            # modify pipeline list
        # Performance Aware
        if N == 3:
            selected_indices_set = sorted(range(len(predecessor_scores)), key=lambda i: predecessor_scores[i],reverse=True)[:K]
        ## Diversity-Aware
        if N == 4:
            selected_pipelines = self.select_pipelines_based_on_diversity(sklearn_pipeline_list, K)
            selected_indices_set = [sklearn_pipeline_list.index(pipeline) for pipeline in selected_pipelines if
                                pipeline in sklearn_pipeline_list]
        ## Performance Cost
        if N == 5:
            perf_cost = []
            for idx, sklearn_pipeline in enumerate(sklearn_pipeline_list):
                request, pipeline, cost = self.estimate_and_add(data_id, sklearn_pipeline)
                # X, y = get_dataset(data_id)
                # artifact_graph, request = extract_artifact_graph(data_id, sklearn_pipeline, features, target)
                #graph_summary = pd.DataFrame({'Pipe': [sklearn_pipeline], 'Cost': [cost],'Performance':[predecessor_scores[i]], 'Chosen': [0]})
                #pipelines_summary = pd.concat([pipelines_summary, graph_summary], ignore_index=True)
                if cost > 0:
                    ratio =  predecessor_scores[idx]/cost
                else:
                    ratio = 100000.0
                perf_cost.append(ratio)
            selected_indices_set = sorted(range(len(perf_cost)), key=lambda i: perf_cost[i],reverse=True)[:K]
        ## Diversity Cost
        if N == 6:
            selected_pipelines = self.select_pipelines_based_on_diversity_cost(data_id, sklearn_pipeline_list, K)
            selected_indices_set = [sklearn_pipeline_list.index(pipeline) for pipeline in selected_pipelines if
                                    pipeline in sklearn_pipeline_list]
        ## Performance Diversity Cost
        if N == 7:
            selected_pipelines = self.select_pipelines_based_on_diversity_cost(data_id, sklearn_pipeline_list, K)
            selected_indices_set = [sklearn_pipeline_list.index(pipeline) for pipeline in selected_pipelines if
                                    pipeline in sklearn_pipeline_list]
            perf_selected_indices_set = sorted(range(len(predecessor_scores)), key=lambda i: predecessor_scores[i],
                                          reverse=True)[:K]
            num_perf_selected = int(0.2 * K)  # 20% from perf_selected_indices_set
            num_selected = K - num_perf_selected  # Remaining 80% from selected_indices_set

            # Select the first num_perf_selected from perf_selected_indices_set
            perf_indices = perf_selected_indices_set[:num_perf_selected]
            # Remove the selected perf_indices from selected_indices_set
            remaining_indices = [idx for idx in selected_indices_set if idx not in perf_indices]
            # Select the first num_selected from the remaining indices
            selected_indices = remaining_indices[:num_selected]

            # Combine the two selections
            selected_indices_set = perf_indices + selected_indices
        ## Performance Diversity Cost 2
        if N == 8:
            selected_indices_set = self.select_pipelines_based_on_performance_diversity_cost(data_id, sklearn_pipeline_list, K, predecessor_scores)
        if N == 9:
            pipelines_summary = pd.DataFrame(columns=['Pipe', 'Cost','Performance'])

            for idx, sklearn_pipeline in enumerate(sklearn_pipeline_list):
                request, pipeline, cost = self.estimate_and_add(data_id, sklearn_pipeline)
                # X, y = get_dataset(data_id)
                # artifact_graph, request = extract_artifact_graph(data_id, sklearn_pipeline, features, target)
                graph_summary = pd.DataFrame({'Pipe': [sklearn_pipeline], 'Cost': [cost],'Performance':[predecessor_scores[idx]]})
                pipelines_summary = pd.concat([pipelines_summary, graph_summary], ignore_index=True)

            sorted_pipelines = pipelines_summary.sort_values(by=['Performance', 'Cost'], ascending=[False, True])
            selected_pipelines_df = sorted_pipelines.head(K)
            selected_indices_set = [sklearn_pipeline_list.index(pipe) for pipe in selected_pipelines_df['Pipe']]
        if N == 10:
            pipelines_summary = pd.DataFrame(columns=['Pipe', 'Cost', 'Performance'])
            for idx, sklearn_pipeline in enumerate(sklearn_pipeline_list):
                request, pipeline, cost = self.estimate_and_add(data_id, sklearn_pipeline)
                # X, y = get_dataset(data_id)
                # artifact_graph, request = extract_artifact_graph(data_id, sklearn_pipeline, features, target)
                graph_summary = pd.DataFrame(
                    {'Pipe': [sklearn_pipeline], 'Cost': [cost], 'Performance': [predecessor_scores[idx]]})
                pipelines_summary = pd.concat([pipelines_summary, graph_summary], ignore_index=True)
            selected_indices_set = self.select_pipelines_pareto_then_performance(pipelines_summary, sklearn_pipeline_list, K)
        if N == 11:
            pipelines_summary = pd.DataFrame(columns=['Pipe', 'Cost', 'Performance'])

            for idx, sklearn_pipeline in enumerate(sklearn_pipeline_list):
                request, pipeline, cost = self.estimate_and_add(data_id, sklearn_pipeline)
                # X, y = get_dataset(data_id)
                # artifact_graph, request = extract_artifact_graph(data_id, sklearn_pipeline, features, target)
                graph_summary = pd.DataFrame(
                    {'Pipe': [sklearn_pipeline], 'Cost': [cost], 'Performance': [predecessor_scores[idx]]})
                pipelines_summary = pd.concat([pipelines_summary, graph_summary], ignore_index=True)
            selected_indices_set = self.select_pipelines_pareto_then_performance_agressive(pipelines_summary, sklearn_pipeline_list)
        if N == 21:
            selected_indices_set =  self.AutoMLselection_lamda(data_id, K, sklearn_pipeline_list, predecessor_scores, l, Utility="performance", timeout=timeout)
        if N == 41:
            selected_indices_set = self.flaml_like_selection(K, sklearn_pipeline_list)
            #lamda
            #selected_indices_set
        return selected_indices_set

    def scale_to_1(self, series):
        x_min = series.min()
        x_max = series.max()
        return ((series - x_min) / (x_max - x_min))

    def scale_to_timeout(self, series, timeout):
        x_min = 0
        x_max = timeout
        return (series / timeout)


    def select_pipelines_based_on_diversity_lamda(self, data_id, pipelines, K, l):
        selected_pipelines = []
        zero_score_pipelines = []  # To store pipelines with a score of zero
        current_diversity = 0  # Initial diversity is zero
        remaining_pipelines = pipelines.copy()  # Start with all pipelines
        pipelines_selected = 0
        pipe_costs = []  # To store the cost of each pipeline

        # Collect the costs for each pipeline
        for pipeline in remaining_pipelines:
            request, pipeline, cost = self.estimate_and_add(data_id, pipeline)
            pipe_costs.append(cost)

        pipe_costs = pd.Series(pipe_costs)  # Convert to Series
        scaled_costs = self.scale_to_1(pipe_costs)  # Scale costs

        while pipelines_selected < K and remaining_pipelines:
            best_ratio = -1
            best_pipeline = None
            best_idx = 0

            # Iterate over a copy of the list to avoid modifying it during iteration
            for idx, pipeline in enumerate(remaining_pipelines.copy()):
                cost = scaled_costs[idx]
                estimator = extract_estimator_from_pipeline(str(pipeline))

                # Temporarily update the estimator counter to calculate the average diversity
                temp_counter = self.evaluated_pipelines.copy()
                temp_counter.update([estimator])

                # Calculate probabilities and entropy for the updated set
                total_estimators = sum(temp_counter.values())
                estimator_probabilities = [count / total_estimators for count in temp_counter.values()]
                new_diversity = compute_entropy(estimator_probabilities)

                # Calculate diversity improvement
                diversity_improvement = new_diversity - current_diversity

                # Track the best pipeline based on the highest diversity improvement / cost ratio
                ratio = (1-l)*diversity_improvement - l*cost

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pipeline = pipeline
                    best_idx = idx

            # Add the best pipeline to the selected sequence
            if best_pipeline is not None:
                selected_pipelines.append(best_pipeline)
                pipelines_selected += 1

                # Update the current diversity and estimator counter
                selected_estimator = extract_estimator_from_pipeline(str(best_pipeline))
                self.evaluated_pipelines.update([selected_estimator])

                # Update the current diversity
                total_estimators = sum(self.evaluated_pipelines.values())
                estimator_probabilities = [count / total_estimators for count in self.evaluated_pipelines.values()]
                current_diversity = compute_entropy(estimator_probabilities)

                # Remove the selected pipeline from the remaining pipelines and its associated cost
                remaining_pipelines.pop(best_idx)
                pipe_costs.pop(best_idx)

        self.save_to_file()  # Save any changes made during the selection process
        return selected_pipelines

    def flaml_like_selection(self, K, sklearn_pipeline_list):
        import pandas as pd

        eci_scores = []

        for idx, pipeline in enumerate(sklearn_pipeline_list):
            learner = extract_estimator_from_pipeline(str(pipeline))

            metrics = self.flaML_metrics.get(learner, {'K0': 0, 'K1': 0, 'K2': 0, 'D1': 0, 'D2': 0})
            K0, K1, K2 = metrics['K0'] or 0, metrics['K1'] or 0, metrics['K2'] or 0
            D1, D2 = metrics['D1'] or 0, metrics['D2'] or 0
            e_star = self.global_best_error  # Use the global best error

            # Compute ECI1 and ECI3
            eci1 = max(K0 - K1, K1 - K2)
            delta = D1 - D2 if D1 > D2 else 1e-6  # Avoid division by zero
            eci3 = ((D1 - e_star) * (K0 - K2)) / delta
            eci = max(eci1, eci3)

            eci_scores.append((idx, eci))

        # Sort pipelines by ECI and select the top K indices
        selected_indices_set = [idx for idx, _ in sorted(eci_scores, key=lambda x: x[1])[:K]]

        return selected_indices_set

    def AutoMLselection_lamda(self,data_id, K, sklearn_pipeline_list, predecessor_scores, l, Utility, timeout=None):
        import pandas as pd
        if Utility == "DIV":
            if isinstance(sklearn_pipeline_list, pd.DataFrame):
                if 'cost' in sklearn_pipeline_list.columns:
                    selected_pipelines = self.select_pipelines_based_on_diversity_lamda_for_df_scaled(data_id, sklearn_pipeline_list, K,
                                                                                        l)
                    selected_indices_set = [
                        sklearn_pipeline_list[sklearn_pipeline_list['pipeline'] == pipeline].index[0]
                        for pipeline in selected_pipelines
                        if pipeline in sklearn_pipeline_list['pipeline'].values
                    ]
            else:
                selected_pipelines = self.select_pipelines_based_on_diversity_lamda( data_id, sklearn_pipeline_list, K, l)
                selected_indices_set = [sklearn_pipeline_list.index(pipeline) for pipeline in selected_pipelines if
                                        pipeline in sklearn_pipeline_list]
        else:
            #self, data_id, timeout, pipelines,predecessor_scores, K, l
            selected_pipelines = self.select_pipelines_based_on_performance_lambda(data_id, timeout, sklearn_pipeline_list, predecessor_scores, K, l)
            selected_indices_set = [sklearn_pipeline_list.index(pipeline) for pipeline in selected_pipelines if
                                    pipeline in sklearn_pipeline_list]
        return selected_indices_set

    def select_pipelines_based_on_performance_lambda(self, data_id, timeout, pipelines, predecessor_scores, K, l):
        """
        Select pipelines based on a weighted combination of performance and scaled cost.

        Args:
            data_id (int): Identifier for the dataset.
            timeout (float): Timeout for scaling the costs.
            pipelines (list): List of pipeline objects.
            predecessor_scores (list): List of performance scores for the pipelines.
            K (int): Number of pipelines to select.
            l (float): Lambda parameter to weight performance and cost.

        Returns:
            list: Selected pipelines.
        """
        selected_pipelines = []  # Store selected pipelines
        pipelines_selected = 0  # Track the number of selected pipelines

        # Create a DataFrame to manage pipelines, costs, and performance
        pipe_costs = []
        for pipeline in pipelines:
            _, _, cost = self.estimate_and_add(data_id, pipeline)
            pipe_costs.append(cost)

        pipelines_df = pd.DataFrame({
            'Pipeline': pipelines,
            'Cost': pipe_costs,
            'Performance': predecessor_scores
        })

        # Scale the costs
        pipelines_df['Scaled_Cost'] = self.scale_to_timeout(pipelines_df['Cost'], timeout)

        # Select pipelines based on the ratio
        while pipelines_selected < K and not pipelines_df.empty:
            # Compute the weighted ratio for each pipeline
            pipelines_df['Ratio'] = (1 - l) * pipelines_df['Performance'] - l * pipelines_df['Scaled_Cost']

            # Select the pipeline with the best ratio
            best_pipeline_idx = pipelines_df['Ratio'].idxmax()
            best_pipeline = pipelines_df.loc[best_pipeline_idx]

            # Add the selected pipeline to the result list
            selected_pipelines.append(best_pipeline['Pipeline'])
            pipelines_selected += 1

            # Remove the selected pipeline from the DataFrame
            pipelines_df = pipelines_df.drop(index=best_pipeline_idx).reset_index(drop=True)

        # Save the selected pipelines to file if needed
        self.save_to_file()

        return selected_pipelines





    def select_pipelines_based_on_performace_lamda_old_2(self, data_id, timeout, pipelines, predecessor_scores, K, l):
        selected_pipelines = []
        current_diversity = 0  # Initial diversity is zero
        remaining_pipelines = pipelines.copy()  # Start with all pipelines
        pipelines_selected = 0

        # Collect the costs for each pipeline
        pipe_costs = []
        for pipeline in remaining_pipelines:
            request, pipeline, cost = self.estimate_and_add(data_id, pipeline)
            pipe_costs.append(cost)

        pipe_costs = pd.Series(pipe_costs)  # Convert to Series
        scaled_costs = self.scale_to_timeout(pipe_costs, timeout)  # Scale costs

        while pipelines_selected < K and remaining_pipelines:
            best_ratio = -1
            best_pipeline = None
            best_idx = None

            # Iterate over the pipelines and calculate ratios
            for idx, pipeline in enumerate(remaining_pipelines):
                try:
                    cost = scaled_costs.iloc[idx]  # Safely access costs using .iloc
                    perf = predecessor_scores[idx]
                    ratio = (1 - l) * perf - l * cost
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_pipeline = pipeline
                        best_idx = idx
                except KeyError:
                    print(f"Index {idx} not found in scaled_costs. Skipping.")
                    continue

            # Add the best pipeline to the selected sequence
            if best_pipeline is not None:
                selected_pipelines.append(best_pipeline)
                pipelines_selected += 1
                # Remove the selected pipeline
                del remaining_pipelines[best_idx]
                scaled_costs = scaled_costs.drop(scaled_costs.index[best_idx]).reset_index(drop=True)

        self.save_to_file()  # Save any changes made during the selection process
        return selected_pipelines

    def select_pipelines_based_on_performace_lamda_old(self, data_id, timeout, pipelines, predecessor_scores, K, l):
        selected_pipelines = []
        zero_score_pipelines = []  # To store pipelines with a score of zero
        current_diversity = 0  # Initial diversity is zero
        remaining_pipelines = pipelines.copy()  # Start with all pipelines
        pipelines_selected = 0
        pipe_costs = []  # To store the cost of each pipeline

        # Collect the costs for each pipeline
        for pipeline in remaining_pipelines:
            request, pipeline, cost = self.estimate_and_add(data_id, pipeline)
            pipe_costs.append(cost)

        pipe_costs = pd.Series(pipe_costs)  # Convert to Series
        scaled_costs = self.scale_to_timeout(pipe_costs, timeout)  # Scale costs

        while pipelines_selected < K and remaining_pipelines:
            best_ratio = -1
            best_pipeline = None
            best_idx = 0

            # Iterate over a copy of the list to avoid modifying it during iteration
            for idx, pipeline in enumerate(remaining_pipelines.copy()):
                cost = scaled_costs[idx]
                perf = predecessor_scores[idx]

                # Temporarily update the estimator counter to calculate the average diversity
                # Calculate probabilities and entropy for the updated set

                # Calculate diversity improvement

                # Track the best pipeline based on the highest diversity improvement / cost ratio
                ratio = (1-l)*perf - l*cost

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pipeline = pipeline
                    best_idx = idx

            # Add the best pipeline to the selected sequence
            if best_pipeline is not None:
                selected_pipelines.append(best_pipeline)
                pipelines_selected += 1
                # Remove the selected pipeline from the remaining pipelines and its associated cost
                remaining_pipelines.pop(best_idx)
                scaled_costs.pop(best_idx)

        self.save_to_file()  # Save any changes made during the selection process
        return selected_pipelines


    def select_pipelines_based_on_diversity_lamda_for_df(self, data_id, pipelines, K, l):
        from collections import Counter

        selected_pipelines = []
        pipelines_selected = 0
        current_diversity = 0
        evaluated_pipelines = Counter(self.evaluated_pipelines)

        # Scale the costs in the 'cost' column
        pipelines = pipelines.copy()
        pipelines['scaled_cost'] = self.scale_to_1(pipelines['cost'])

        while pipelines_selected < K and not pipelines.empty:
            # Precompute diversity improvement for all rows
            def compute_diversity_improvement(row):
                estimator = extract_estimator_from_pipeline(str(row['pipeline']))
                temp_counter = evaluated_pipelines.copy()
                temp_counter.update([estimator])

                # Calculate new diversity
                total_estimators = sum(temp_counter.values())
                probabilities = [count / total_estimators for count in temp_counter.values()]
                new_diversity = compute_entropy(probabilities)

                # Diversity improvement
                return new_diversity - current_diversity

            pipelines['diversity_improvement'] = pipelines.apply(compute_diversity_improvement, axis=1)
            pipelines['ratio'] = pipelines['diversity_improvement'] - l * pipelines['scaled_cost']

            # Select the pipeline with the best ratio
            best_idx = pipelines['ratio'].idxmax()
            best_row = pipelines.loc[best_idx]

            # Update selected pipelines
            selected_pipelines.append(best_row['pipeline'])
            pipelines_selected += 1

            # Update diversity and evaluated pipelines
            best_estimator = extract_estimator_from_pipeline(str(best_row['pipeline']))
            evaluated_pipelines.update([best_estimator])

            total_estimators = sum(evaluated_pipelines.values())
            probabilities = [count / total_estimators for count in evaluated_pipelines.values()]
            current_diversity = compute_entropy(probabilities)

            # Remove the selected pipeline
            pipelines = pipelines.drop(index=best_idx)

        self.evaluated_pipelines = evaluated_pipelines
        self.save_to_file()
        return selected_pipelines

    def select_pipelines_based_on_diversity_lamda_for_df_scaled(self, data_id, pipelines, K, l):
        from collections import Counter

        selected_pipelines = []
        pipelines_selected = 0
        current_diversity = 0
        evaluated_pipelines = Counter(self.evaluated_pipelines)

        # Scale the costs in the 'cost' column
        pipelines = pipelines.copy()
        pipelines['scaled_cost'] = self.scale_to_1(pipelines['cost'])

        while pipelines_selected < K and not pipelines.empty:
            # Precompute diversity improvement for all rows
            def compute_diversity_improvement(row):
                estimator = extract_estimator_from_pipeline(str(row['pipeline']))
                temp_counter = evaluated_pipelines.copy()
                temp_counter.update([estimator])

                # Calculate new diversity
                total_estimators = sum(temp_counter.values())
                probabilities = [count / total_estimators for count in temp_counter.values()]
                new_diversity = compute_entropy(probabilities)

                # Diversity improvement
                return new_diversity - current_diversity

            pipelines['diversity_improvement'] = pipelines.apply(compute_diversity_improvement, axis=1)

            # Normalize diversity improvement to [0, 1]
            min_div = pipelines['diversity_improvement'].min()
            max_div = pipelines['diversity_improvement'].max()

            if max_div > min_div:  # Avoid division by zero
                pipelines['scaled_diversity_improvement'] = (pipelines['diversity_improvement'] - min_div) / (
                            max_div - min_div)
            else:
                pipelines['scaled_diversity_improvement'] = 0  # If all values are the same

            # Compute the ratio
            pipelines['ratio'] = (1 - l) * pipelines['scaled_diversity_improvement'] - l * pipelines['scaled_cost']

            # Select the pipeline with the best ratio
            best_idx = pipelines['ratio'].idxmax()
            best_row = pipelines.loc[best_idx]

            # Update selected pipelines
            selected_pipelines.append(best_row['pipeline'])
            pipelines_selected += 1

            # Update diversity and evaluated pipelines
            best_estimator = extract_estimator_from_pipeline(str(best_row['pipeline']))
            evaluated_pipelines.update([best_estimator])

            total_estimators = sum(evaluated_pipelines.values())
            probabilities = [count / total_estimators for count in evaluated_pipelines.values()]
            current_diversity = compute_entropy(probabilities)

            # Remove the selected pipeline
            pipelines = pipelines.drop(index=best_idx)

        self.evaluated_pipelines = evaluated_pipelines
        self.save_to_file()
        return selected_pipelines


    def select_pipelines_based_on_diversity_lamda_for_df_slow(self, data_id, pipelines, K, l):
        selected_pipelines = []
        zero_score_pipelines = []  # To store pipelines with a score of zero
        current_diversity = 0  # Initial diversity is zero
        remaining_pipelines = pipelines.copy()  # Start with all pipelines
        pipelines_selected = 0

        # Scale the costs in the 'cost' column
        remaining_pipelines['scaled_cost'] = self.scale_to_1(remaining_pipelines['cost'])

        while pipelines_selected < K and not remaining_pipelines.empty:
            best_ratio = -1
            best_pipeline = None
            best_idx = None

            for idx, row in remaining_pipelines.iterrows():
                pipeline = row['pipeline']
                cost = row['scaled_cost']
                estimator = extract_estimator_from_pipeline(str(pipeline))

                # Temporarily update the estimator counter to calculate the average diversity
                temp_counter = self.evaluated_pipelines.copy()
                temp_counter.update([estimator])

                # Calculate probabilities and entropy for the updated set
                total_estimators = sum(temp_counter.values())
                estimator_probabilities = [count / total_estimators for count in temp_counter.values()]
                new_diversity = compute_entropy(estimator_probabilities)

                # Calculate diversity improvement
                diversity_improvement = new_diversity - current_diversity

                # Track the best pipeline based on the highest diversity improvement / cost ratio
                ratio = (1 - l) * diversity_improvement - l * cost

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pipeline = pipeline
                    best_idx = idx

            # Add the best pipeline to the selected sequence
            if best_pipeline is not None:
                selected_pipelines.append(best_pipeline)
                pipelines_selected += 1

                # Update the current diversity and estimator counter
                selected_estimator = extract_estimator_from_pipeline(str(best_pipeline))
                self.evaluated_pipelines.update([selected_estimator])

                # Update the current diversity
                total_estimators = sum(self.evaluated_pipelines.values())
                estimator_probabilities = [count / total_estimators for count in self.evaluated_pipelines.values()]
                current_diversity = compute_entropy(estimator_probabilities)

                # Remove the selected pipeline from the remaining pipelines
                remaining_pipelines = remaining_pipelines.drop(index=best_idx)

        self.save_to_file()  # Save any changes made during the selection process
        return selected_pipelines