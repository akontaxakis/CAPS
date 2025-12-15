import os
import time

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from caps.components.Cost_estimator.AutoML_data_manager.data_manager import DataManager
from caps.components.Cost_estimator.pipeline_estimator import estimate_cost
from caps.components.Cost_estimator.util import error_metrics, plot_predictions
from caps.components.lib import extract_artifact_graph

plt.switch_backend('Agg')
os.environ["LOKY_MAX_CPU_COUNT"] = "6"

if __name__ == '__main__':
    #iris = load_iris()
    test_df = pd.DataFrame(columns=['Original', 'Predicted'])
    data_id = "dilbert"
    #regression_model = "ExtraTrees"
    regression_model = "Gradient Boosting"
    datasets = ["sylvine", "philippine", "jannis", "christine", "dilbert", "fabert", "albert", "digits", "fabert"]
    datasets_path = r"C:\Users\adoko\PycharmProjects\autoPipe\autoML\datasets"

    data = DataManager(datasets[0], datasets_path, replace_missing=True,
                     verbose=3)
    X = data.data['X_train']
    y = data.data['Y_train']

    pipelines = [
        Pipeline([
        ('DT', DecisionTreeClassifier(criterion="gini", max_depth=10, min_samples_split=3, min_samples_leaf=10))
    ]) ,Pipeline([
        ('scaler', StandardScaler()),
        ('DT', DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_split=3, min_samples_leaf=5))
    ]), Pipeline([
        ('scaler', StandardScaler()),
        ('DT', DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_split=12, min_samples_leaf=10))
    ]), Pipeline([
        ('scaler', StandardScaler()),
        ('DT', DecisionTreeClassifier(criterion="gini", max_depth=8, min_samples_split=13, min_samples_leaf=10))
    ]), Pipeline([
         ('PCA', PCA(svd_solver='randomized',
        iterated_power= 2)),
        ('DT', DecisionTreeClassifier(criterion="gini", max_depth=2, min_samples_split=3, min_samples_leaf=3))
    ]), Pipeline([
        ('scaler', StandardScaler()),
        ('DT', DecisionTreeClassifier(criterion="gini", max_depth=2, min_samples_split=3, min_samples_leaf=10))
    ]), Pipeline([
        ('PCA', PCA( svd_solver='randomized',
        iterated_power= 10)),
        ('DT', DecisionTreeClassifier(criterion="gini", max_depth=2, min_samples_split=3, min_samples_leaf=10))
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
        ('KN', KNeighborsClassifier( n_neighbors=  20,
        weights= "uniform",
        p= 1))
    ]),Pipeline([
        ('KN', KNeighborsClassifier( n_neighbors=  10,
        weights= "uniform",
        p= 2))
    ]),
        Pipeline([
            ('KN', GradientBoostingClassifier(n_estimators= 100,
        learning_rate=  0.5,
        max_depth= 5,
        min_samples_split= 10,
        min_samples_leaf= 10,
        subsample=0.1,
        max_features= 1))
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
        artifact_graph, request = extract_artifact_graph(data_id, pipe,X,y)

        pipeline_graph, cost = estimate_cost(artifact_graph, regression_model)

        print("estimated cost:" + str(cost))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        start_time = time.time()
        pipe.fit(X_train, y_train)
        prediction = pipe.predict(X_test)
        end_time = time.time()
        sum_time = end_time - start_time

        print("true cost:" + str(sum_time))
        df = pd.DataFrame({'Original': [sum_time], 'Predicted': [cost]})
        test_df = pd.concat([test_df, df], ignore_index=True)
    # Calculate error metrics
    test_df = error_metrics(test_df)
    # Plot predictions
    plot_predictions("full execution time of pipelines ","regression", test_df)
    print(test_df)