import os
import time

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from Cost_estimator.AutoML_data_manager.data_manager import DataManager
from Cost_estimator.pipeline_estimator import estimate_cost
from Cost_estimator.util import error_metrics, plot_predictions, random_shape
from HYPPO.components.HistoryGraph import HistoryGraph
from HYPPO.components.parser.parser import extract_artifact_graph, get_dataset

plt.switch_backend('Agg')
os.environ["LOKY_MAX_CPU_COUNT"] = "6"

if __name__ == '__main__':

    #iris = load_iris()
    test_df = pd.DataFrame(columns=['Original', 'Predicted'])

    pipelines_summary_true = pd.DataFrame(columns=['Pipe', 'Request','Chosen'])
    pipelines_summary_est = pd.DataFrame(columns=['Pipe', 'Request','Chosen'])

    data_id = "jannis"
    Hyppo_true = HistoryGraph("History_true")
    Hyppo_est = HistoryGraph("History_est")

    Hyppo_true.add_dataset_split(data_id,0.3)
    Hyppo_est.add_dataset_split(data_id,0.3)
    #regression_model = "ExtraTrees"
    regression_model = "Gradient Boosting"
    #data_id = ["jannis"]
    K = 5

    pipelines = [
        Pipeline([
        ('DT', DecisionTreeClassifier(criterion="gini", max_depth=10, min_samples_split=3, min_samples_leaf=10))
    ]) , Pipeline([
        ('scaler', StandardScaler()),
        ('DT', DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_split=12, min_samples_leaf=10))
    ]), Pipeline([
        ('scaler', StandardScaler()),
        ('DT', DecisionTreeClassifier(criterion="gini", max_depth=8, min_samples_split=13, min_samples_leaf=10))
    ]), Pipeline([
        ('scaler', StandardScaler()), ('PCA', PCA(svd_solver='randomized',
        iterated_power= 10)),
        ('DT', DecisionTreeClassifier(criterion="gini", max_depth=10, min_samples_split=3, min_samples_leaf=10))
    ]), Pipeline([
        ('scaler', StandardScaler()),
        ('DT', DecisionTreeClassifier(criterion="gini", max_depth=10, min_samples_split=3, min_samples_leaf=10))
    ]), Pipeline([
        ('PCA', PCA( svd_solver='randomized',
        iterated_power= 10)),
        ('DT', DecisionTreeClassifier(criterion="gini", max_depth=10, min_samples_split=3, min_samples_leaf=10))
    ]), Pipeline([
        ('Norm', Normalizer(norm='l1')),
        ('DT', ExtraTreesClassifier(n_estimators= 100,
        criterion= "gini",
        max_features= 1.0,
        min_samples_split= 10,
        min_samples_leaf= 10,
        bootstrap=False))
    ]),Pipeline([
        ('KN', ExtraTreesClassifier(n_estimators= 100,
        criterion= "gini",
        max_features= 1.0,
        min_samples_split= 10,
        min_samples_leaf= 10,
        bootstrap=False))
    ]),Pipeline([
        ('scaler', StandardScaler()),
        ('KN', KNeighborsClassifier())
    ]),Pipeline([
        ('KN', KNeighborsClassifier( n_neighbors=  50,
        weights= "uniform",
        p= 1))
    ]),Pipeline([
        ('KN', KNeighborsClassifier( n_neighbors=  30,
        weights= "uniform",
        p= 2))
    ])
    ]

    '''
    pipelines = [
        Pipeline([
            ('scaler', StandardScaler()),
            ('DT', DecisionTreeClassifier(criterion="gini", max_depth=8, min_samples_split=13, min_samples_leaf=10))
        ]), Pipeline([
            ('scaler', StandardScaler()),
            ('DT', DecisionTreeClassifier(criterion="gini", max_depth=10, min_samples_split=3, min_samples_leaf=10))
        ]), Pipeline([
            ('PCA', PCA(svd_solver='randomized',
                        iterated_power=10)),
            ('DT', DecisionTreeClassifier(criterion="gini", max_depth=10, min_samples_split=3, min_samples_leaf=10))
        ]), Pipeline([
            ('Norm', Normalizer(norm='l1')),
            ('DT', ExtraTreesClassifier(n_estimators=100,
                                        criterion="gini",
                                        max_features=1.0,
                                        min_samples_split=10,
                                        min_samples_leaf=10,
                                        bootstrap=False))
        ])]
    '''
    for pipe in pipelines:
        Hyppo_true.execute_and_add(data_id, pipe)
        Hyppo_est.estimate_and_add(data_id, pipe)
        X, y = get_dataset(data_id)
        artifact_graph, request = extract_artifact_graph(data_id, pipe,X,y)
        graph_summary = pd.DataFrame({'Pipe': [pipe], 'Request': [request], 'Chosen': [0]})
        pipelines_summary_true = pd.concat([pipelines_summary_true, graph_summary], ignore_index=True)
        pipelines_summary_est = pd.concat([pipelines_summary_est, graph_summary],ignore_index=True)

    execution_graph, pipelines, total_cost = Hyppo_est.prune(K, pipelines_summary_est)
    print(pipelines)
    print(total_cost)

    execution_graph, pipelines, total_cost = Hyppo_true.prune(K, pipelines_summary_true)
    print(pipelines)
    print(total_cost)



    #Hyppo_true.visualize(mode='use_alias')
    #Hyppo_est.visualize(mode='use_alias')