import os
import time
from types import NoneType

import networkx as nx
import pandas as pd

from caps.components.Cost_estimator.util import flatten_parameters_column


def cost_estimator(opt, task, params, n_features, n_samples,regression_model):
    model_dir = r"models"
    operator_name = opt.__class__.__name__
    cost = 0
    features = []
    # Search for .plk file in the directory that contains the operator and model names
    model_file = None
    if task == 'ftransform' or task == 'tetransform':
        task ='transform'
    for file in os.listdir(model_dir):
        if file.endswith(".pkl") and operator_name in file and task in file:
            model_file = os.path.join(model_dir, file)
            break

    if model_file:
        #print(f"Found model file: {model_file}")
        # Load the model file and perform cost estimation (example with joblib)
        import joblib
        model = joblib.load(model_file)
        # Example of how you might use the model for cost estimation
        # This part will depend on the structure of your model and how it is used

        features.append({
            'n_features': n_features,
            'n_samples': n_samples,
            'parameters': params
        })
        df = pd.DataFrame(features)
        df = flatten_parameters_column(df)
        df = pd.get_dummies(df)
        df = adjust_features(df, model.feature_names_in_)
        df = df.dropna(axis=1, how='all')
        cost = model.predict(df)[0]
        if cost < 0:
            cost =0

    return cost

def estimate_cost_time(pipeline_graph, regression_model, history = None):
    cost = 0
    graph_check_time = 0
    topological_order = list(nx.topological_sort(pipeline_graph))

    # Iterate over the nodes in topological order
    for artifact in topological_order:
        a_samples = pipeline_graph.nodes[artifact].get("samples", 1)
        a_features = pipeline_graph.nodes[artifact].get("features", 1)

        # Using out_edges method
        for task in pipeline_graph.out_edges(artifact):

            output_artifact = task[1]
            # Get the value of the specific attribute
            opt = pipeline_graph.get_edge_data(task[0], task[1]).get("function", None)
            if opt is not None:
                params = opt.get_params()
            else:
                params = None
            type = pipeline_graph.get_edge_data(task[0], task[1]).get("type", None)
            n_features, n_samples = shape_estimator(opt,type, params,a_features, a_samples)
            a_samples = pipeline_graph.nodes[output_artifact].get("samples", 1)
            a_features = pipeline_graph.nodes[output_artifact].get("features", 1)
            c_s= max(a_samples, n_samples)
            c_f = max(a_features, n_features)
            pipeline_graph.nodes[output_artifact]["samples"] = c_s
            pipeline_graph.nodes[output_artifact]["features"] = c_f
            a_samples = pipeline_graph.nodes[artifact].get("samples", 1)
            a_features = pipeline_graph.nodes[artifact].get("features", 1)
            s_time = time.time()
            graph = True
            est_cost =0
            if history is not None and history.has_edge(task[0], task[1]):
                est_cost = pipeline_graph[task[0]][task[1]].get("execution_time", None)
                graph =False
            graph_check_time = graph_check_time + (time.time() - s_time)
            if graph or est_cost == 0:
                est_cost = cost_estimator(opt, type, params, a_features, a_samples, regression_model)
            pipeline_graph[task[0]][task[1]]["execution_time"] = est_cost
            pipeline_graph[task[0]][task[1]]["weight"] = est_cost
            cost = cost + est_cost
    return pipeline_graph, cost, graph_check_time

def estimate_cost(pipeline_graph, regression_model, history = None):
    cost = 0
    topological_order = list(nx.topological_sort(pipeline_graph))

    # Iterate over the nodes in topological order
    for artifact in topological_order:
        a_samples = pipeline_graph.nodes[artifact].get("samples", 1)
        a_features = pipeline_graph.nodes[artifact].get("features", 1)

        # Using out_edges method
        for task in pipeline_graph.out_edges(artifact):

            output_artifact = task[1]
            # Get the value of the specific attribute
            opt = pipeline_graph.get_edge_data(task[0], task[1]).get("function", None)
            if opt is not None:
                params = opt.get_params()
            else:
                params = None
            type = pipeline_graph.get_edge_data(task[0], task[1]).get("type", None)
            n_features, n_samples = shape_estimator(opt,type, params,a_features, a_samples)
            a_samples = pipeline_graph.nodes[output_artifact].get("samples", 1)
            a_features = pipeline_graph.nodes[output_artifact].get("features", 1)
            c_s= max(a_samples, n_samples)
            c_f = max(a_features, n_features)
            pipeline_graph.nodes[output_artifact]["samples"] = c_s
            pipeline_graph.nodes[output_artifact]["features"] = c_f
            a_samples = pipeline_graph.nodes[artifact].get("samples", 1)
            a_features = pipeline_graph.nodes[artifact].get("features", 1)
            start_time = time.time()
            graph = True
            if history is not None and history.has_edge(task[0], task[1]):
                est_cost = pipeline_graph[task[0]][task[1]].get("execution_time", None)
                graph =False
            if graph:
                est_cost = cost_estimator(opt, type, params, a_features, a_samples, regression_model)
            pipeline_graph[task[0]][task[1]]["execution_time"] = est_cost
            pipeline_graph[task[0]][task[1]]["weight"] = est_cost
            cost = cost + est_cost
    return pipeline_graph, cost


def adjust_features(input_df, trained_features):
    adjusted_df = input_df.reindex(columns=trained_features, fill_value=0)
    return adjusted_df

def shape_estimator(opt, task,params, n_features, n_samples):
    if task == "fit":
        return 1, 1
    if task == "predict":
        return 1, n_samples

    return n_features, n_samples



