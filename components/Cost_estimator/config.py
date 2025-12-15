import numpy as np
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import FastICA, PCA
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFwe, SelectPercentile, RFE, VarianceThreshold, SelectFromModel
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.preprocessing import Binarizer, MaxAbsScaler, MinMaxScaler, Normalizer, PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, SGDClassifier

# Define classifiers and their configurations
from tpot.builtins import ZeroCount, OneHotEncoder
from xgboost import XGBClassifier

operators_config = {
    'sklearn.naive_bayes.GaussianNB': GaussianNB,
    'sklearn.naive_bayes.BernoulliNB': BernoulliNB,
    'sklearn.naive_bayes.MultinomialNB': MultinomialNB,
    'sklearn.tree.DecisionTreeClassifier': DecisionTreeClassifier,
    'sklearn.ensemble.ExtraTreesClassifier': ExtraTreesClassifier,
    'sklearn.ensemble.RandomForestClassifier': RandomForestClassifier,
    'sklearn.ensemble.GradientBoostingClassifier':GradientBoostingClassifier,
    'sklearn.neighbors.KNeighborsClassifier': KNeighborsClassifier,
    'sklearn.svm.LinearSVC':LinearSVC,
    'sklearn.linear_model.SGDClassifier': SGDClassifier,
    'xgboost.XGBClassifier': XGBClassifier,
    #preprocessing
    'sklearn.preprocessing.Binarizer':Binarizer,
    'sklearn.decomposition.FastICA':FastICA,
    'sklearn.cluster.FeatureAgglomeration':FeatureAgglomeration,
    'sklearn.preprocessing.MaxAbsScaler':MaxAbsScaler,
    'sklearn.preprocessing.MinMaxScaler':MinMaxScaler,
    'sklearn.preprocessing.Normalizer':Normalizer,
    'sklearn.kernel_approximation.Nystroem':Nystroem,
    'sklearn.decomposition.PCA':PCA,
    'sklearn.preprocessing.PolynomialFeatures':PolynomialFeatures,
    'sklearn.kernel_approximation.RBFSampler':RBFSampler,
    'sklearn.preprocessing.StandardScaler':StandardScaler,
    'tpot.builtins.ZeroCount':ZeroCount,
    'tpot.builtins.OneHotEncoder':OneHotEncoder,
    # Selectors
    'sklearn.feature_selection.SelectFwe':SelectFwe,
    'sklearn.feature_selection.SelectPercentile': SelectPercentile,
    'sklearn.feature_selection.VarianceThreshold':VarianceThreshold,
    'sklearn.feature_selection.RFE':RFE,
    'sklearn.feature_selection.SelectFromModel':SelectFromModel

}

config_params = {
    # Classifiers
    'sklearn.naive_bayes.GaussianNB': {
    },
    'sklearn.naive_bayes.BernoulliNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },
    'sklearn.naive_bayes.MultinomialNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },
    'sklearn.tree.DecisionTreeClassifier': {
        'criterion': ["gini", "entropy"],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },
    'sklearn.ensemble.ExtraTreesClassifier': {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },
    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf':  range(1, 21),
        'bootstrap': [True, False]
    },
    'sklearn.ensemble.GradientBoostingClassifier': {
        'n_estimators': [100],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05)
    },
    'sklearn.neighbors.KNeighborsClassifier': {
        'n_neighbors': range(1, 101),
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },
    'sklearn.svm.LinearSVC': {
        'penalty': ["l2"],
        #'penalty':  ["l1", "l2"],
        #'loss': ["hinge", "squared_hinge"],

        'loss': ["squared_hinge"],
        #'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]
    },
    'sklearn.linear_model.LogisticRegression': {
        'penalty': ["l1", "l2"],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'dual': [True, False]
    },
    'xgboost.XGBClassifier': {
        'n_estimators': [100],
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'n_jobs': [1],
        'verbosity': [0]
    },
    'sklearn.linear_model.SGDClassifier': {
        'loss': ['log_loss', 'hinge', 'modified_huber', 'squared_hinge', 'perceptron'],
        'penalty': ['elasticnet'],
        'alpha': [0.0, 0.01, 0.001],
        'learning_rate': ['invscaling', 'constant'],
        'fit_intercept': [True, False],
        'l1_ratio': [0.25, 0.0, 1.0, 0.75, 0.5],
        'eta0': [0.1, 1.0, 0.01],
        'power_t': [0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0]
    },

    'sklearn.neural_network.MLPClassifier': {
        'alpha': [1e-4, 1e-3, 1e-2, 1e-1],
        'learning_rate_init': [1e-3, 1e-2, 1e-1, 0.5, 1.]
    },

    # Preprocesssors
    'sklearn.preprocessing.Binarizer': {
        'threshold': np.arange(0.0, 1.01, 0.05)
    },
    'sklearn.decomposition.FastICA': {
        'tol': np.arange(0.0, 1.01, 0.05)
    },
    'sklearn.cluster.FeatureAgglomeration': {
        'linkage': ['ward', 'complete', 'average'],
        'metric': ['euclidean']
        #'metric': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']

    },
    'sklearn.preprocessing.MaxAbsScaler': {
    },
    'sklearn.preprocessing.MinMaxScaler': {
    },
    'sklearn.preprocessing.Normalizer': {
        'norm': ['l1', 'l2', 'max']
    },
    'sklearn.kernel_approximation.Nystroem': {
        'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
        'gamma': np.arange(0.0, 1.01, 0.05),
        'n_components': range(1, 11)
    },
    'sklearn.decomposition.PCA': {
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 11)
    },
    'sklearn.preprocessing.PolynomialFeatures': {
        'degree': [2],
        'include_bias': [False],
        'interaction_only': [False]
    },
    'sklearn.kernel_approximation.RBFSampler': {
        'gamma': np.arange(0.0, 1.01, 0.05)
    },
    'sklearn.preprocessing.RobustScaler': {
    },
    'sklearn.preprocessing.StandardScaler': {
    },
    'tpot.builtins.ZeroCount': {
    },
    'tpot.builtins.OneHotEncoder': {
        'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25],
        'sparse': [False],
        'threshold': [10]
    },

    # Selectors
    'sklearn.feature_selection.SelectFwe': {
        'alpha': np.arange(0, 0.05, 0.001)
        #'score_func': {
        #    'sklearn.feature_selection.f_classif': None
        #}
    },
    'sklearn.feature_selection.SelectPercentile': {
        'percentile': range(1, 100)
        #'score_func': {
        #    'sklearn.feature_selection.f_classif': None
        #}
    },
    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    },
    'sklearn.feature_selection.RFE': {
        'step': np.arange(0.05, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': [100],
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    },
    'sklearn.feature_selection.SelectFromModel': {
        'threshold': np.arange(0, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': [100],
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    }



}

