import time

import numpy as np
from ConfigSpace import Configuration
from imblearn.pipeline import Pipeline
from parso.normalizer import Normalizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Binarizer, MaxAbsScaler, MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier



def train_with_budget_full(
    config: Configuration,
    budget: float,
    X,
    y,
    seed: int = 0
) -> float:
    import numpy as np

    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    from sklearn.preprocessing import (
        Binarizer, MaxAbsScaler, MinMaxScaler,
        Normalizer, RobustScaler, StandardScaler
    )
    from sklearn.decomposition import PCA, FastICA
    from sklearn.cluster import FeatureAgglomeration
    from sklearn.kernel_approximation import Nystroem, RBFSampler
    from sklearn.preprocessing import PolynomialFeatures

    from sklearn.feature_selection import (
        SelectFwe, SelectPercentile, VarianceThreshold,
        RFE, SelectFromModel
    )
    from sklearn.feature_selection import f_classif

    from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier,
        GradientBoostingClassifier
    )
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier

    from tpot.builtins import ZeroCount

    try:
        # ==================================================
        # Preprocessing
        # ==================================================

        prep_name = config['preprocessing']
        if prep_name == 'None':
            preprocessing = None
        elif prep_name == 'sklearn.preprocessing.Binarizer':
            preprocessing = Binarizer(threshold=config['Binarizer__threshold'])
        elif prep_name == 'sklearn.preprocessing.MaxAbsScaler':
            preprocessing = MaxAbsScaler()
        elif prep_name == 'sklearn.preprocessing.MinMaxScaler':
            preprocessing = MinMaxScaler()
        elif prep_name == 'sklearn.preprocessing.Normalizer':
            preprocessing = Normalizer(norm=config['Normalizer__norm'])
        elif prep_name == 'sklearn.preprocessing.RobustScaler':
            preprocessing = RobustScaler()
        elif prep_name == 'sklearn.preprocessing.StandardScaler':
            preprocessing = StandardScaler()
        else:
            raise ValueError(prep_name)

        # ==================================================
        # Feature engineering
        # ==================================================

        fe_name = config['feature_engineering']
        if fe_name == 'None':
            fe = None
        elif fe_name == 'sklearn.decomposition.FastICA':
            fe = FastICA(tol=config['FastICA__tol'], random_state=seed)
        elif fe_name == 'sklearn.cluster.FeatureAgglomeration':
            fe = FeatureAgglomeration(
                linkage=config['FeatureAgglomeration__linkage'],
                affinity=config['FeatureAgglomeration__affinity']
            )
        elif fe_name == 'sklearn.kernel_approximation.Nystroem':
            fe = Nystroem(
                kernel=config['Nystroem__kernel'],
                gamma=config['Nystroem__gamma'],
                n_components=config['Nystroem__n_components'],
                random_state=seed
            )
        elif fe_name == 'sklearn.decomposition.PCA':
            fe = PCA(
                svd_solver=config['PCA__svd_solver'],
                iterated_power=config['PCA__iterated_power'],
                random_state=seed
            )
        elif fe_name == 'sklearn.preprocessing.PolynomialFeatures':
            fe = PolynomialFeatures(
                degree=config['PolynomialFeatures__degree'],
                include_bias=config['PolynomialFeatures__include_bias'],
                interaction_only=config['PolynomialFeatures__interaction_only']
            )
        elif fe_name == 'sklearn.kernel_approximation.RBFSampler':
            fe = RBFSampler(
                gamma=config['RBFSampler__gamma'],
                random_state=seed
            )
        elif fe_name == 'tpot.builtins.ZeroCount':
            fe = ZeroCount()
        elif fe_name == 'sklearn.feature_selection.SelectFwe':
            fe = SelectFwe(
                alpha=config['SelectFwe__alpha'],
                score_func=f_classif
            )
        elif fe_name == 'sklearn.feature_selection.SelectPercentile':
            fe = SelectPercentile(
                percentile=config['SelectPercentile__percentile'],
                score_func=f_classif
            )
        elif fe_name == 'sklearn.feature_selection.VarianceThreshold':
            fe = VarianceThreshold(
                threshold=config['VarianceThreshold__threshold']
            )
        elif fe_name == 'sklearn.feature_selection.RFE':
            fe = RFE(
                estimator=ExtraTreesClassifier(
                    n_estimators=100,
                    criterion='gini',
                    max_features=0.5,
                    random_state=seed
                ),
                step=config['RFE__step']
            )
        elif fe_name == 'sklearn.feature_selection.SelectFromModel':
            fe = SelectFromModel(
                estimator=ExtraTreesClassifier(
                    n_estimators=100,
                    criterion='gini',
                    max_features=0.5,
                    random_state=seed
                ),
                threshold=config['SelectFromModel__threshold']
            )
        else:
            raise ValueError(fe_name)

        # ==================================================
        # Classifier
        # ==================================================

        clf_name = config['classifier']
        if clf_name == 'sklearn.naive_bayes.GaussianNB':
            clf = GaussianNB()
        elif clf_name == 'sklearn.naive_bayes.BernoulliNB':
            clf = BernoulliNB(
                alpha=config['BernoulliNB__alpha'],
                fit_prior=config['BernoulliNB__fit_prior']
            )
        elif clf_name == 'sklearn.naive_bayes.MultinomialNB':
            clf = MultinomialNB(
                alpha=config['MultinomialNB__alpha'],
                fit_prior=config['MultinomialNB__fit_prior']
            )
        elif clf_name == 'sklearn.tree.DecisionTreeClassifier':
            clf = DecisionTreeClassifier(
                criterion=config['DecisionTreeClassifier__criterion'],
                max_depth=config['DecisionTreeClassifier__max_depth'],
                min_samples_split=config['DecisionTreeClassifier__min_samples_split'],
                min_samples_leaf=config['DecisionTreeClassifier__min_samples_leaf'],
                random_state=seed
            )
        elif clf_name == 'sklearn.ensemble.RandomForestClassifier':
            clf = RandomForestClassifier(
                n_estimators=100,
                max_features=config['RandomForestClassifier__max_features'],
                min_samples_split=config['RandomForestClassifier__min_samples_split'],
                min_samples_leaf=config['RandomForestClassifier__min_samples_leaf'],
                criterion=config['RandomForestClassifier__criterion'],
                bootstrap=config['RandomForestClassifier__bootstrap'],
                random_state=seed
            )
        elif clf_name == 'sklearn.ensemble.ExtraTreesClassifier':
            clf = ExtraTreesClassifier(
                n_estimators=100,
                max_features=config['ExtraTreesClassifier__max_features'],
                min_samples_split=config['ExtraTreesClassifier__min_samples_split'],
                min_samples_leaf=config['ExtraTreesClassifier__min_samples_leaf'],
                criterion=config['ExtraTreesClassifier__criterion'],
                bootstrap=config['ExtraTreesClassifier__bootstrap'],
                random_state=seed
            )
        elif clf_name == 'sklearn.ensemble.GradientBoostingClassifier':
            clf = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=config['GradientBoostingClassifier__learning_rate'],
                max_depth=config['GradientBoostingClassifier__max_depth'],
                min_samples_split=config['GradientBoostingClassifier__min_samples_split'],
                min_samples_leaf=config['GradientBoostingClassifier__min_samples_leaf'],
                subsample=config['GradientBoostingClassifier__subsample'],
                max_features=config['GradientBoostingClassifier__max_features'],
                random_state=seed
            )
        elif clf_name == 'sklearn.neighbors.KNeighborsClassifier':
            clf = KNeighborsClassifier(
                n_neighbors=config['KNeighborsClassifier__n_neighbors'],
                weights=config['KNeighborsClassifier__weights'],
                p=config['KNeighborsClassifier__p']
            )
        elif clf_name == 'sklearn.svm.LinearSVC':
            clf = LinearSVC(
                C=config['LinearSVC__C'],
                loss=config['LinearSVC__loss'],
                penalty=config['LinearSVC__penalty'],
                dual=config['LinearSVC__dual'],
                tol=config['LinearSVC__tol'],
                random_state=seed
            )
        elif clf_name == 'sklearn.linear_model.LogisticRegression':
            clf = LogisticRegression(
                C=config['LogisticRegression__C'],
                penalty=config['LogisticRegression__penalty'],
                dual=config['LogisticRegression__dual'],
                max_iter=1000,
                random_state=seed
            )
        elif clf_name == 'sklearn.linear_model.SGDClassifier':
            clf = SGDClassifier(
                loss=config['SGDClassifier__loss'],
                alpha=config['SGDClassifier__alpha'],
                learning_rate=config['SGDClassifier__learning_rate'],
                fit_intercept=config['SGDClassifier__fit_intercept'],
                l1_ratio=config['SGDClassifier__l1_ratio'],
                eta0=config['SGDClassifier__eta0'],
                power_t=config['SGDClassifier__power_t'],
                random_state=seed
            )
        elif clf_name == 'sklearn.neural_network.MLPClassifier':
            clf = MLPClassifier(
                alpha=config['MLPClassifier__alpha'],
                learning_rate_init=config['MLPClassifier__learning_rate_init'],
                random_state=seed,
                max_iter=200
            )
        elif clf_name == 'xgboost.XGBClassifier':
            clf = XGBClassifier(
                n_estimators=100,
                max_depth=config['XGBClassifier__max_depth'],
                learning_rate=config['XGBClassifier__learning_rate'],
                subsample=config['XGBClassifier__subsample'],
                min_child_weight=config['XGBClassifier__min_child_weight'],
                verbosity=0,
                n_jobs=1,
                random_state=seed,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            raise ValueError(clf_name)

        # ==================================================
        # Pipeline
        # ==================================================

        steps = []
        if preprocessing is not None:
            steps.append(('preprocessing', preprocessing))
        if fe is not None:
            steps.append(('feature_engineering', fe))
        steps.append(('classifier', clf))

        pipeline = Pipeline(steps)

        # ==================================================
        # Budget handling
        # ==================================================

        frac = float(np.clip(budget, 0.05, 1.0))
        n = len(y)
        rng = np.random.RandomState(seed)
        m = max(50, int(frac * n))
        idx = rng.permutation(n)[:m]
        Xb, yb = X[idx], y[idx]

        cv = StratifiedKFold(n_splits=2, shuffle=False)
        scores = cross_val_score(pipeline, Xb, yb, cv=cv)

        return 1.0 - float(np.mean(scores))

    except Exception:
        return 1.0



def train_full(
    config: Configuration,
    X,
    y,
    seed: int = 0
) -> float:
    import numpy as np

    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    from sklearn.preprocessing import (
        Binarizer, MaxAbsScaler, MinMaxScaler,
        Normalizer, RobustScaler, StandardScaler
    )
    from sklearn.decomposition import PCA, FastICA
    from sklearn.cluster import FeatureAgglomeration
    from sklearn.kernel_approximation import Nystroem, RBFSampler
    from sklearn.preprocessing import PolynomialFeatures

    from sklearn.feature_selection import (
        SelectFwe, SelectPercentile, VarianceThreshold,
        RFE, SelectFromModel
    )
    from sklearn.feature_selection import f_classif

    from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier,
        GradientBoostingClassifier
    )
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier

    from tpot.builtins import ZeroCount

    try:
        # ==================================================
        # Preprocessing
        # ==================================================

        prep_name = config['preprocessing']
        if prep_name == 'None':
            preprocessing = None
        elif prep_name == 'sklearn.preprocessing.Binarizer':
            preprocessing = Binarizer(threshold=config['Binarizer__threshold'])
        elif prep_name == 'sklearn.preprocessing.MaxAbsScaler':
            preprocessing = MaxAbsScaler()
        elif prep_name == 'sklearn.preprocessing.MinMaxScaler':
            preprocessing = MinMaxScaler()
        elif prep_name == 'sklearn.preprocessing.Normalizer':
            preprocessing = Normalizer(norm=config['Normalizer__norm'])
        elif prep_name == 'sklearn.preprocessing.RobustScaler':
            preprocessing = RobustScaler()
        elif prep_name == 'sklearn.preprocessing.StandardScaler':
            preprocessing = StandardScaler()
        else:
            raise ValueError(prep_name)

        # ==================================================
        # Feature engineering
        # ==================================================

        fe_name = config['feature_engineering']
        if fe_name == 'None':
            fe = None
        elif fe_name == 'sklearn.decomposition.FastICA':
            fe = FastICA(tol=config['FastICA__tol'], random_state=seed)
        elif fe_name == 'sklearn.cluster.FeatureAgglomeration':
            fe = FeatureAgglomeration(
                linkage=config['FeatureAgglomeration__linkage'],
                affinity=config['FeatureAgglomeration__affinity']
            )
        elif fe_name == 'sklearn.kernel_approximation.Nystroem':
            fe = Nystroem(
                kernel=config['Nystroem__kernel'],
                gamma=config['Nystroem__gamma'],
                n_components=config['Nystroem__n_components'],
                random_state=seed
            )
        elif fe_name == 'sklearn.decomposition.PCA':
            fe = PCA(
                svd_solver=config['PCA__svd_solver'],
                iterated_power=config['PCA__iterated_power'],
                random_state=seed
            )
        elif fe_name == 'sklearn.preprocessing.PolynomialFeatures':
            fe = PolynomialFeatures(
                degree=config['PolynomialFeatures__degree'],
                include_bias=config['PolynomialFeatures__include_bias'],
                interaction_only=config['PolynomialFeatures__interaction_only']
            )
        elif fe_name == 'sklearn.kernel_approximation.RBFSampler':
            fe = RBFSampler(
                gamma=config['RBFSampler__gamma'],
                random_state=seed
            )
        elif fe_name == 'tpot.builtins.ZeroCount':
            fe = ZeroCount()
        elif fe_name == 'sklearn.feature_selection.SelectFwe':
            fe = SelectFwe(
                alpha=config['SelectFwe__alpha'],
                score_func=f_classif
            )
        elif fe_name == 'sklearn.feature_selection.SelectPercentile':
            fe = SelectPercentile(
                percentile=config['SelectPercentile__percentile'],
                score_func=f_classif
            )
        elif fe_name == 'sklearn.feature_selection.VarianceThreshold':
            fe = VarianceThreshold(
                threshold=config['VarianceThreshold__threshold']
            )
        elif fe_name == 'sklearn.feature_selection.RFE':
            fe = RFE(
                estimator=ExtraTreesClassifier(
                    n_estimators=100,
                    criterion='gini',
                    max_features=0.5,
                    random_state=seed
                ),
                step=config['RFE__step']
            )
        elif fe_name == 'sklearn.feature_selection.SelectFromModel':
            fe = SelectFromModel(
                estimator=ExtraTreesClassifier(
                    n_estimators=100,
                    criterion='gini',
                    max_features=0.5,
                    random_state=seed
                ),
                threshold=config['SelectFromModel__threshold']
            )
        else:
            raise ValueError(fe_name)

        # ==================================================
        # Classifier
        # ==================================================

        clf_name = config['classifier']
        if clf_name == 'sklearn.naive_bayes.GaussianNB':
            clf = GaussianNB()
        elif clf_name == 'sklearn.naive_bayes.BernoulliNB':
            clf = BernoulliNB(
                alpha=config['BernoulliNB__alpha'],
                fit_prior=config['BernoulliNB__fit_prior']
            )
        elif clf_name == 'sklearn.naive_bayes.MultinomialNB':
            clf = MultinomialNB(
                alpha=config['MultinomialNB__alpha'],
                fit_prior=config['MultinomialNB__fit_prior']
            )
        elif clf_name == 'sklearn.tree.DecisionTreeClassifier':
            clf = DecisionTreeClassifier(
                criterion=config['DecisionTreeClassifier__criterion'],
                max_depth=config['DecisionTreeClassifier__max_depth'],
                min_samples_split=config['DecisionTreeClassifier__min_samples_split'],
                min_samples_leaf=config['DecisionTreeClassifier__min_samples_leaf'],
                random_state=seed
            )
        elif clf_name == 'sklearn.ensemble.RandomForestClassifier':
            clf = RandomForestClassifier(
                n_estimators=100,
                max_features=config['RandomForestClassifier__max_features'],
                min_samples_split=config['RandomForestClassifier__min_samples_split'],
                min_samples_leaf=config['RandomForestClassifier__min_samples_leaf'],
                criterion=config['RandomForestClassifier__criterion'],
                bootstrap=config['RandomForestClassifier__bootstrap'],
                random_state=seed
            )
        elif clf_name == 'sklearn.ensemble.ExtraTreesClassifier':
            clf = ExtraTreesClassifier(
                n_estimators=100,
                max_features=config['ExtraTreesClassifier__max_features'],
                min_samples_split=config['ExtraTreesClassifier__min_samples_split'],
                min_samples_leaf=config['ExtraTreesClassifier__min_samples_leaf'],
                criterion=config['ExtraTreesClassifier__criterion'],
                bootstrap=config['ExtraTreesClassifier__bootstrap'],
                random_state=seed
            )
        elif clf_name == 'sklearn.ensemble.GradientBoostingClassifier':
            clf = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=config['GradientBoostingClassifier__learning_rate'],
                max_depth=config['GradientBoostingClassifier__max_depth'],
                min_samples_split=config['GradientBoostingClassifier__min_samples_split'],
                min_samples_leaf=config['GradientBoostingClassifier__min_samples_leaf'],
                subsample=config['GradientBoostingClassifier__subsample'],
                max_features=config['GradientBoostingClassifier__max_features'],
                random_state=seed
            )
        elif clf_name == 'sklearn.neighbors.KNeighborsClassifier':
            clf = KNeighborsClassifier(
                n_neighbors=config['KNeighborsClassifier__n_neighbors'],
                weights=config['KNeighborsClassifier__weights'],
                p=config['KNeighborsClassifier__p']
            )
        elif clf_name == 'sklearn.svm.LinearSVC':
            clf = LinearSVC(
                C=config['LinearSVC__C'],
                loss=config['LinearSVC__loss'],
                penalty=config['LinearSVC__penalty'],
                dual=config['LinearSVC__dual'],
                tol=config['LinearSVC__tol'],
                random_state=seed
            )
        elif clf_name == 'sklearn.linear_model.LogisticRegression':
            clf = LogisticRegression(
                C=config['LogisticRegression__C'],
                penalty=config['LogisticRegression__penalty'],
                dual=config['LogisticRegression__dual'],
                max_iter=1000,
                random_state=seed
            )
        elif clf_name == 'sklearn.linear_model.SGDClassifier':
            clf = SGDClassifier(
                loss=config['SGDClassifier__loss'],
                alpha=config['SGDClassifier__alpha'],
                learning_rate=config['SGDClassifier__learning_rate'],
                fit_intercept=config['SGDClassifier__fit_intercept'],
                l1_ratio=config['SGDClassifier__l1_ratio'],
                eta0=config['SGDClassifier__eta0'],
                power_t=config['SGDClassifier__power_t'],
                random_state=seed
            )
        elif clf_name == 'sklearn.neural_network.MLPClassifier':
            clf = MLPClassifier(
                alpha=config['MLPClassifier__alpha'],
                learning_rate_init=config['MLPClassifier__learning_rate_init'],
                random_state=seed,
                max_iter=200
            )
        elif clf_name == 'xgboost.XGBClassifier':
            clf = XGBClassifier(
                n_estimators=100,
                max_depth=config['XGBClassifier__max_depth'],
                learning_rate=config['XGBClassifier__learning_rate'],
                subsample=config['XGBClassifier__subsample'],
                min_child_weight=config['XGBClassifier__min_child_weight'],
                verbosity=0,
                n_jobs=1,
                random_state=seed,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            raise ValueError(clf_name)

        # ==================================================
        # Pipeline + evaluation
        # ==================================================

        steps = []
        if preprocessing is not None:
            steps.append(('preprocessing', preprocessing))
        if fe is not None:
            steps.append(('feature_engineering', fe))
        steps.append(('classifier', clf))

        pipeline = Pipeline(steps)

        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
        scores = cross_val_score(pipeline, X, y, cv=cv)

        return 1.0 - float(np.mean(scores))

    except Exception:
        return 1.0




def train_with_budget(config: Configuration, budget: float,X,y, seed: int = 0) -> float:
    """Train a model based on the configuration provided and return the validation error."""

    # Preprocessor configuration
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
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    # Construct the pipeline with the preprocessor and classifier
    steps = [('preprocessor', preprocessor), ('model', model)]
    pipeline = Pipeline(steps)
    # subsample by budget
    frac = float(np.clip(budget, 0.05, 1.0))
    n = len(y)
    rng = np.random.RandomState(seed)
    m = max(50, int(frac * n))
    idx = rng.permutation(n)[:m]
    Xb, yb = X[idx], y[idx]
    # Perform Cross-Validation
    cv = StratifiedKFold(n_splits=2, shuffle=False)
    scores = cross_val_score(pipeline, Xb, yb, cv=cv)

    # Return the validation error (1 - mean accuracy)
    return 1 - np.mean(scores)


def train_without_budget(config: Configuration,X,y, seed: int = 0) -> float:
    """Train a model based on the configuration provided and return the validation error."""

    # Preprocessor configuration
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
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    # Construct the pipeline with the preprocessor and classifier
    steps = [('preprocessor', preprocessor), ('model', model)]
    pipeline = Pipeline(steps)

    # Perform Cross-Validation
    cv = StratifiedKFold(n_splits=2, shuffle=False)
    scores = cross_val_score(pipeline, X, y, cv=cv)

    # Return the validation error (1 - mean accuracy)
    return 1 - np.mean(scores)

def train_quality_cost(config: Configuration,X,y, seed: int = 0) -> tuple[float, float]:
    """Train a model and return both quality (error rate) and cost (runtime)."""
    start_time = time.time()

    # Build pipeline (same as before)
    preprocessor_name = config['preprocessor']
    classifier_name = config['classifier']

    # Define preprocessor
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

    # Define classifier
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
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    # Construct the pipeline with the preprocessor and classifier
    steps = [('preprocessor', preprocessor), ('model', model)]
    pipeline = Pipeline(steps)

    # Perform Cross-Validation
    cv = StratifiedKFold(n_splits=2, shuffle=False)
    scores = cross_val_score(pipeline, X, y, cv=cv)

    # Compute quality (validation error)
    quality = 1 - np.mean(scores)

    # Compute cost (runtime)
    runtime = time.time() - start_time

    return {"quality": quality, "cost": runtime}