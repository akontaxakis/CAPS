import numpy as np

from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import FastICA, PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier
)
from sklearn.feature_selection import (
    SelectFwe,
    SelectPercentile,
    RFE,
    VarianceThreshold,
    SelectFromModel
)
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.preprocessing import (
    Binarizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PolynomialFeatures,
    StandardScaler
)
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

from tpot.builtins import ZeroCount, OneHotEncoder
from xgboost import XGBClassifier

# ============================================================
# Operator registry
# ============================================================

operators_config = {
    'sklearn.naive_bayes.GaussianNB': GaussianNB,
    'sklearn.naive_bayes.BernoulliNB': BernoulliNB,
    'sklearn.naive_bayes.MultinomialNB': MultinomialNB,
    'sklearn.tree.DecisionTreeClassifier': DecisionTreeClassifier,
    'sklearn.ensemble.ExtraTreesClassifier': ExtraTreesClassifier,
    'sklearn.ensemble.RandomForestClassifier': RandomForestClassifier,
    'sklearn.ensemble.GradientBoostingClassifier': GradientBoostingClassifier,
    'sklearn.neighbors.KNeighborsClassifier': KNeighborsClassifier,
    'sklearn.svm.LinearSVC': LinearSVC,
    'sklearn.linear_model.SGDClassifier': SGDClassifier,
    'xgboost.XGBClassifier': XGBClassifier,

    # preprocessors
    'sklearn.preprocessing.Binarizer': Binarizer,
    'sklearn.decomposition.FastICA': FastICA,
    'sklearn.cluster.FeatureAgglomeration': FeatureAgglomeration,
    'sklearn.preprocessing.MaxAbsScaler': MaxAbsScaler,
    'sklearn.preprocessing.MinMaxScaler': MinMaxScaler,
    'sklearn.preprocessing.Normalizer': Normalizer,
    'sklearn.kernel_approximation.Nystroem': Nystroem,
    'sklearn.decomposition.PCA': PCA,
    'sklearn.preprocessing.PolynomialFeatures': PolynomialFeatures,
    'sklearn.kernel_approximation.RBFSampler': RBFSampler,
    'sklearn.preprocessing.StandardScaler': StandardScaler,
    'tpot.builtins.ZeroCount': ZeroCount,
    'tpot.builtins.OneHotEncoder': OneHotEncoder,

    # selectors
    'sklearn.feature_selection.SelectFwe': SelectFwe,
    'sklearn.feature_selection.SelectPercentile': SelectPercentile,
    'sklearn.feature_selection.VarianceThreshold': VarianceThreshold,
    'sklearn.feature_selection.RFE': RFE,
    'sklearn.feature_selection.SelectFromModel': SelectFromModel,
}

# ============================================================
# Base estimators for meta-selectors (CRITICAL FIX)
# ============================================================

BASE_ESTIMATORS = {
    'ExtraTrees': ExtraTreesClassifier(
        n_estimators=100,
        criterion='gini',
        max_features=0.5,
        random_state=42,
        n_jobs=1
    )
}

# ============================================================
# Hyperparameter spaces (ParameterGrid-safe)
# ============================================================

config_params = {

    # ----------------- classifiers -----------------

    'sklearn.naive_bayes.GaussianNB': {},

    'sklearn.naive_bayes.BernoulliNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },

    'sklearn.naive_bayes.MultinomialNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },

    'sklearn.tree.DecisionTreeClassifier': {
        'criterion': ['gini', 'entropy'],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },

    'sklearn.ensemble.ExtraTreesClassifier': {
        'n_estimators': [100],
        'criterion': ['gini', 'entropy'],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': [100],
        'criterion': ['gini', 'entropy'],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
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
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    },

    'sklearn.svm.LinearSVC': {
        'loss': ['squared_hinge'],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10.]
    },

    'sklearn.linear_model.SGDClassifier': {
        'loss': ['log_loss', 'hinge', 'modified_huber'],
        'penalty': ['elasticnet'],
        'alpha': [0.0, 0.01, 0.001],
        'learning_rate': ['invscaling', 'constant'],
        'l1_ratio': [0.0, 0.5, 1.0],
    },

    'xgboost.XGBClassifier': {
        'n_estimators': [100],
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5],
        'subsample': np.arange(0.1, 1.01, 0.1),
        'verbosity': [0],
        'n_jobs': [1],
    },

    # ----------------- preprocessors -----------------

    'sklearn.preprocessing.Binarizer': {
        'threshold': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.decomposition.FastICA': {
        'tol': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.cluster.FeatureAgglomeration': {
        'linkage': ['ward', 'complete', 'average'],
        'metric': ['euclidean']
    },

    'sklearn.preprocessing.MaxAbsScaler': {},
    'sklearn.preprocessing.MinMaxScaler': {},

    'sklearn.preprocessing.Normalizer': {
        'norm': ['l1', 'l2', 'max']
    },

    'sklearn.kernel_approximation.Nystroem': {
        'kernel': ['rbf', 'cosine'],
        'gamma': np.arange(0.0, 1.01, 0.05),
        'n_components': range(1, 11)
    },

    'sklearn.decomposition.PCA': {
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 11)
    },

    'sklearn.preprocessing.PolynomialFeatures': {
        'degree': [2],
        'include_bias': [False]
    },

    'sklearn.kernel_approximation.RBFSampler': {
        'gamma': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.preprocessing.StandardScaler': {},

    'tpot.builtins.ZeroCount': {},

    'tpot.builtins.OneHotEncoder': {
        'minimum_fraction': [0.05, 0.1, 0.2],
        'sparse': [False],
        'threshold': [10]
    },

    # ----------------- selectors  -----------------

    'sklearn.feature_selection.SelectFwe': {
        'alpha': np.arange(0, 0.05, 0.001)
    },

    'sklearn.feature_selection.SelectPercentile': {
        'percentile': range(1, 100)
    },

    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': [0.0001, 0.001, 0.01, 0.1]
    },

    'sklearn.feature_selection.RFE': {
        'estimator': [BASE_ESTIMATORS['ExtraTrees']],
        'step': np.arange(0.05, 1.01, 0.05)
    },

    'sklearn.feature_selection.SelectFromModel': {
        'estimator': [BASE_ESTIMATORS['ExtraTrees']],
        'threshold': np.arange(0.0, 1.01, 0.05)
    },
}
