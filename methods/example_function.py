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
    config,
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
        Normalizer, RobustScaler, StandardScaler,
        PolynomialFeatures
    )
    from sklearn.decomposition import PCA, FastICA
    from sklearn.cluster import FeatureAgglomeration
    from sklearn.kernel_approximation import Nystroem, RBFSampler

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
        # ================= Preprocessing =================
        prep = config["preprocessing"]
        if prep == "None":
            preprocessing = None
        elif prep == "sklearn.preprocessing.Binarizer":
            preprocessing = Binarizer(
                threshold=config.get("Binarizer__threshold", 0.0)
            )
        elif prep == "sklearn.preprocessing.MaxAbsScaler":
            preprocessing = MaxAbsScaler()
        elif prep == "sklearn.preprocessing.MinMaxScaler":
            preprocessing = MinMaxScaler()
        elif prep == "sklearn.preprocessing.Normalizer":
            preprocessing = Normalizer(
                norm=config.get("Normalizer__norm", "l2")
            )
        elif prep == "sklearn.preprocessing.RobustScaler":
            preprocessing = RobustScaler()
        elif prep == "sklearn.preprocessing.StandardScaler":
            preprocessing = StandardScaler()
        else:
            raise ValueError(prep)

        # ================= Feature engineering =================
        fe_name = config["feature_engineering"]
        if fe_name == "None":
            fe = None

        elif fe_name == "sklearn.decomposition.FastICA":
            fe = FastICA(
                tol=config.get("FastICA__tol", 0.0001),
                random_state=seed
            )

        elif fe_name == "sklearn.cluster.FeatureAgglomeration":
            try:
                fe = FeatureAgglomeration(
                    linkage=config.get("FeatureAgglomeration__linkage", "ward"),
                    metric=config.get("FeatureAgglomeration__affinity", "euclidean")
                )
            except TypeError:
                fe = FeatureAgglomeration(
                    linkage=config.get("FeatureAgglomeration__linkage", "ward"),
                    affinity=config.get("FeatureAgglomeration__affinity", "euclidean")
                )

        elif fe_name == "sklearn.kernel_approximation.Nystroem":
            fe = Nystroem(
                kernel=config.get("Nystroem__kernel", "rbf"),
                gamma=config.get("Nystroem__gamma", 0.1),
                n_components=config.get("Nystroem__n_components", 100),
                random_state=seed
            )

        elif fe_name == "sklearn.decomposition.PCA":
            fe = PCA(
                svd_solver=config.get("PCA__svd_solver", "randomized"),
                iterated_power=config.get("PCA__iterated_power", 1),
                random_state=seed
            )

        elif fe_name == "sklearn.preprocessing.PolynomialFeatures":
            fe = PolynomialFeatures(
                degree=config.get("PolynomialFeatures__degree", 2),
                include_bias=config.get("PolynomialFeatures__include_bias", False),
                interaction_only=config.get("PolynomialFeatures__interaction_only", False)
            )

        elif fe_name == "sklearn.kernel_approximation.RBFSampler":
            fe = RBFSampler(
                gamma=config.get("RBFSampler__gamma", 0.1),
                random_state=seed
            )

        elif fe_name == "tpot.builtins.ZeroCount":
            fe = ZeroCount()

        elif fe_name == "sklearn.feature_selection.SelectFwe":
            fe = SelectFwe(
                alpha=config.get("SelectFwe__alpha", 0.05),
                score_func=f_classif
            )

        elif fe_name == "sklearn.feature_selection.SelectPercentile":
            fe = SelectPercentile(
                percentile=config.get("SelectPercentile__percentile", 50),
                score_func=f_classif
            )

        elif fe_name == "sklearn.feature_selection.VarianceThreshold":
            fe = VarianceThreshold(
                threshold=config.get("VarianceThreshold__threshold", 0.0)
            )

        elif fe_name == "sklearn.feature_selection.RFE":
            fe = RFE(
                estimator=ExtraTreesClassifier(
                    n_estimators=100,
                    criterion="gini",
                    max_features=0.5,
                    random_state=seed
                ),
                step=config.get("RFE__step", 0.1)
            )

        elif fe_name == "sklearn.feature_selection.SelectFromModel":
            fe = SelectFromModel(
                estimator=ExtraTreesClassifier(
                    n_estimators=100,
                    criterion="gini",
                    max_features=0.5,
                    random_state=seed
                ),
                threshold=config.get("SelectFromModel__threshold", 0.0)
            )
        else:
            raise ValueError(fe_name)

        # ================= Classifier =================
        cls = config["classifier"]

        if cls == "sklearn.naive_bayes.GaussianNB":
            clf = GaussianNB()

        elif cls == "sklearn.naive_bayes.BernoulliNB":
            clf = BernoulliNB(
                alpha=config.get("BernoulliNB__alpha", 1.0),
                fit_prior=config.get("BernoulliNB__fit_prior", True)
            )

        elif cls == "sklearn.naive_bayes.MultinomialNB":
            clf = MultinomialNB(
                alpha=config.get("MultinomialNB__alpha", 1.0),
                fit_prior=config.get("MultinomialNB__fit_prior", True)
            )

        elif cls == "sklearn.tree.DecisionTreeClassifier":
            clf = DecisionTreeClassifier(
                criterion=config.get("DecisionTreeClassifier__criterion", "gini"),
                max_depth=config.get("DecisionTreeClassifier__max_depth", None),
                min_samples_split=config.get("DecisionTreeClassifier__min_samples_split", 2),
                min_samples_leaf=config.get("DecisionTreeClassifier__min_samples_leaf", 1),
                random_state=seed
            )

        elif cls == "sklearn.ensemble.RandomForestClassifier":
            clf = RandomForestClassifier(
                n_estimators=100,
                max_features=config.get("RandomForestClassifier__max_features", "sqrt"),
                min_samples_split=config.get("RandomForestClassifier__min_samples_split", 2),
                min_samples_leaf=config.get("RandomForestClassifier__min_samples_leaf", 1),
                criterion=config.get("RandomForestClassifier__criterion", "gini"),
                bootstrap=config.get("RandomForestClassifier__bootstrap", False),
                random_state=seed
            )

        elif cls == "sklearn.ensemble.ExtraTreesClassifier":
            clf = ExtraTreesClassifier(
                n_estimators=100,
                max_features=config.get("ExtraTreesClassifier__max_features", "sqrt"),
                min_samples_split=config.get("ExtraTreesClassifier__min_samples_split", 2),
                min_samples_leaf=config.get("ExtraTreesClassifier__min_samples_leaf", 1),
                criterion=config.get("ExtraTreesClassifier__criterion", "gini"),
                bootstrap=config.get("ExtraTreesClassifier__bootstrap", False),
                random_state=seed
            )

        elif cls == "sklearn.ensemble.GradientBoostingClassifier":
            clf = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=config.get("GradientBoostingClassifier__learning_rate", 0.1),
                max_depth=config.get("GradientBoostingClassifier__max_depth", 3),
                min_samples_split=config.get("GradientBoostingClassifier__min_samples_split", 2),
                min_samples_leaf=config.get("GradientBoostingClassifier__min_samples_leaf", 1),
                subsample=config.get("GradientBoostingClassifier__subsample", 1.0),
                max_features=config.get("GradientBoostingClassifier__max_features", None),
                random_state=seed
            )

        elif cls == "sklearn.neighbors.KNeighborsClassifier":
            clf = KNeighborsClassifier(
                n_neighbors=config.get("KNeighborsClassifier__n_neighbors", 5),
                weights=config.get("KNeighborsClassifier__weights", "uniform"),
                p=config.get("KNeighborsClassifier__p", 2)
            )

        elif cls == "sklearn.svm.LinearSVC":
            clf = LinearSVC(
                C=config.get("LinearSVC__C", 1.0),
                loss=config.get("LinearSVC__loss", "squared_hinge"),
                penalty=config.get("LinearSVC__penalty", "l2"),
                dual=config.get("LinearSVC__dual", True),
                tol=config.get("LinearSVC__tol", 1e-4),
                random_state=seed
            )

        elif cls == "sklearn.linear_model.LogisticRegression":
            clf = LogisticRegression(
                C=config.get("LogisticRegression__C", 1.0),
                penalty=config.get("LogisticRegression__penalty", "l2"),
                dual=config.get("LogisticRegression__dual", False),
                max_iter=1000,
                random_state=seed
            )

        elif cls == "sklearn.linear_model.SGDClassifier":
            clf = SGDClassifier(
                loss=config.get("SGDClassifier__loss", "hinge"),
                alpha=config.get("SGDClassifier__alpha", 0.0001),
                learning_rate=config.get("SGDClassifier__learning_rate", "optimal"),
                fit_intercept=config.get("SGDClassifier__fit_intercept", True),
                l1_ratio=config.get("SGDClassifier__l1_ratio", 0.15),
                eta0=config.get("SGDClassifier__eta0", 0.01),
                power_t=config.get("SGDClassifier__power_t", 0.5),
                random_state=seed
            )

        elif cls == "sklearn.neural_network.MLPClassifier":
            clf = MLPClassifier(
                alpha=config.get("MLPClassifier__alpha", 0.0001),
                learning_rate_init=config.get("MLPClassifier__learning_rate_init", 0.001),
                max_iter=200,
                random_state=seed
            )

        elif cls == "xgboost.XGBClassifier":
            clf = XGBClassifier(
                n_estimators=100,
                max_depth=config.get("XGBClassifier__max_depth", 6),
                learning_rate=config.get("XGBClassifier__learning_rate", 0.3),
                subsample=config.get("XGBClassifier__subsample", 1.0),
                min_child_weight=config.get("XGBClassifier__min_child_weight", 1),
                verbosity=0,
                n_jobs=1,
                random_state=seed,
                use_label_encoder=False,
                eval_metric="logloss"
            )
        else:
            raise ValueError(cls)

        # ================= Pipeline =================
        steps = []
        if preprocessing is not None:
            steps.append(("preprocessing", preprocessing))
        if fe is not None:
            steps.append(("feature_engineering", fe))
        steps.append(("classifier", clf))

        pipeline = Pipeline(steps)

        # ================= Budget =================
        frac = float(np.clip(budget, 0.05, 1.0))
        n = len(y)
        rng = np.random.RandomState(seed)
        m = max(50, int(frac * n))
        idx = rng.permutation(n)[:m]
        Xb, yb = X[idx], y[idx]

        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
        scores = cross_val_score(pipeline, Xb, yb, cv=cv)

        return 1.0 - float(np.mean(scores))

    except Exception:
        return 1.0




def train_full(
    config,
    X,
    y,
    seed: int = 0
) -> float:
    import numpy as np

    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    from sklearn.preprocessing import (
        Binarizer, MaxAbsScaler, MinMaxScaler,
        Normalizer, RobustScaler, StandardScaler,
        PolynomialFeatures
    )
    from sklearn.decomposition import PCA, FastICA
    from sklearn.cluster import FeatureAgglomeration
    from sklearn.kernel_approximation import Nystroem, RBFSampler

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
        prep = config["preprocessing"]
        if prep == "None":
            preprocessing = None
        elif prep == "sklearn.preprocessing.Binarizer":
            preprocessing = Binarizer(
                threshold=config.get("Binarizer__threshold", 0.0)
            )
        elif prep == "sklearn.preprocessing.MaxAbsScaler":
            preprocessing = MaxAbsScaler()
        elif prep == "sklearn.preprocessing.MinMaxScaler":
            preprocessing = MinMaxScaler()
        elif prep == "sklearn.preprocessing.Normalizer":
            preprocessing = Normalizer(
                norm=config.get("Normalizer__norm", "l2")
            )
        elif prep == "sklearn.preprocessing.RobustScaler":
            preprocessing = RobustScaler()
        elif prep == "sklearn.preprocessing.StandardScaler":
            preprocessing = StandardScaler()
        else:
            raise ValueError(prep)

        # ==================================================
        # Feature engineering
        # ==================================================
        fe_name = config["feature_engineering"]
        if fe_name == "None":
            fe = None

        elif fe_name == "sklearn.decomposition.FastICA":
            fe = FastICA(
                tol=config.get("FastICA__tol", 0.0001),
                random_state=seed
            )

        elif fe_name == "sklearn.cluster.FeatureAgglomeration":
            try:
                fe = FeatureAgglomeration(
                    linkage=config.get("FeatureAgglomeration__linkage", "ward"),
                    metric=config.get("FeatureAgglomeration__affinity", "euclidean")
                )
            except TypeError:
                fe = FeatureAgglomeration(
                    linkage=config.get("FeatureAgglomeration__linkage", "ward"),
                    affinity=config.get("FeatureAgglomeration__affinity", "euclidean")
                )

        elif fe_name == "sklearn.kernel_approximation.Nystroem":
            fe = Nystroem(
                kernel=config.get("Nystroem__kernel", "rbf"),
                gamma=config.get("Nystroem__gamma", 0.1),
                n_components=config.get("Nystroem__n_components", 100),
                random_state=seed
            )

        elif fe_name == "sklearn.decomposition.PCA":
            fe = PCA(
                svd_solver=config.get("PCA__svd_solver", "randomized"),
                iterated_power=config.get("PCA__iterated_power", 1),
                random_state=seed
            )

        elif fe_name == "sklearn.preprocessing.PolynomialFeatures":
            fe = PolynomialFeatures(
                degree=config.get("PolynomialFeatures__degree", 2),
                include_bias=config.get("PolynomialFeatures__include_bias", False),
                interaction_only=config.get("PolynomialFeatures__interaction_only", False)
            )

        elif fe_name == "sklearn.kernel_approximation.RBFSampler":
            fe = RBFSampler(
                gamma=config.get("RBFSampler__gamma", 0.1),
                random_state=seed
            )

        elif fe_name == "tpot.builtins.ZeroCount":
            fe = ZeroCount()

        elif fe_name == "sklearn.feature_selection.SelectFwe":
            fe = SelectFwe(
                alpha=config.get("SelectFwe__alpha", 0.05),
                score_func=f_classif
            )

        elif fe_name == "sklearn.feature_selection.SelectPercentile":
            fe = SelectPercentile(
                percentile=config.get("SelectPercentile__percentile", 50),
                score_func=f_classif
            )

        elif fe_name == "sklearn.feature_selection.VarianceThreshold":
            fe = VarianceThreshold(
                threshold=config.get("VarianceThreshold__threshold", 0.0)
            )

        elif fe_name == "sklearn.feature_selection.RFE":
            fe = RFE(
                estimator=ExtraTreesClassifier(
                    n_estimators=100,
                    criterion="gini",
                    max_features=0.5,
                    random_state=seed
                ),
                step=config.get("RFE__step", 0.1)
            )

        elif fe_name == "sklearn.feature_selection.SelectFromModel":
            fe = SelectFromModel(
                estimator=ExtraTreesClassifier(
                    n_estimators=100,
                    criterion="gini",
                    max_features=0.5,
                    random_state=seed
                ),
                threshold=config.get("SelectFromModel__threshold", 0.0)
            )
        else:
            raise ValueError(fe_name)

        # ==================================================
        # Classifier (ALL SAFE)
        # ==================================================
        cls = config["classifier"]

        if cls == "sklearn.naive_bayes.GaussianNB":
            clf = GaussianNB()

        elif cls == "sklearn.naive_bayes.BernoulliNB":
            clf = BernoulliNB(
                alpha=config.get("BernoulliNB__alpha", 1.0),
                fit_prior=config.get("BernoulliNB__fit_prior", True)
            )

        elif cls == "sklearn.naive_bayes.MultinomialNB":
            clf = MultinomialNB(
                alpha=config.get("MultinomialNB__alpha", 1.0),
                fit_prior=config.get("MultinomialNB__fit_prior", True)
            )

        elif cls == "sklearn.tree.DecisionTreeClassifier":
            clf = DecisionTreeClassifier(
                criterion=config.get("DecisionTreeClassifier__criterion", "gini"),
                max_depth=config.get("DecisionTreeClassifier__max_depth", None),
                min_samples_split=config.get("DecisionTreeClassifier__min_samples_split", 2),
                min_samples_leaf=config.get("DecisionTreeClassifier__min_samples_leaf", 1),
                random_state=seed
            )

        elif cls == "sklearn.ensemble.RandomForestClassifier":
            clf = RandomForestClassifier(
                n_estimators=100,
                max_features=config.get("RandomForestClassifier__max_features", "sqrt"),
                min_samples_split=config.get("RandomForestClassifier__min_samples_split", 2),
                min_samples_leaf=config.get("RandomForestClassifier__min_samples_leaf", 1),
                criterion=config.get("RandomForestClassifier__criterion", "gini"),
                bootstrap=config.get("RandomForestClassifier__bootstrap", False),
                random_state=seed
            )

        elif cls == "sklearn.ensemble.ExtraTreesClassifier":
            clf = ExtraTreesClassifier(
                n_estimators=100,
                max_features=config.get("ExtraTreesClassifier__max_features", "sqrt"),
                min_samples_split=config.get("ExtraTreesClassifier__min_samples_split", 2),
                min_samples_leaf=config.get("ExtraTreesClassifier__min_samples_leaf", 1),
                criterion=config.get("ExtraTreesClassifier__criterion", "gini"),
                bootstrap=config.get("ExtraTreesClassifier__bootstrap", False),
                random_state=seed
            )

        elif cls == "sklearn.ensemble.GradientBoostingClassifier":
            clf = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=config.get("GradientBoostingClassifier__learning_rate", 0.1),
                max_depth=config.get("GradientBoostingClassifier__max_depth", 3),
                min_samples_split=config.get("GradientBoostingClassifier__min_samples_split", 2),
                min_samples_leaf=config.get("GradientBoostingClassifier__min_samples_leaf", 1),
                subsample=config.get("GradientBoostingClassifier__subsample", 1.0),
                max_features=config.get("GradientBoostingClassifier__max_features", None),
                random_state=seed
            )

        elif cls == "sklearn.neighbors.KNeighborsClassifier":
            clf = KNeighborsClassifier(
                n_neighbors=config.get("KNeighborsClassifier__n_neighbors", 5),
                weights=config.get("KNeighborsClassifier__weights", "uniform"),
                p=config.get("KNeighborsClassifier__p", 2)
            )

        elif cls == "sklearn.svm.LinearSVC":
            clf = LinearSVC(
                C=config.get("LinearSVC__C", 1.0),
                loss=config.get("LinearSVC__loss", "squared_hinge"),
                penalty=config.get("LinearSVC__penalty", "l2"),
                dual=config.get("LinearSVC__dual", True),
                tol=config.get("LinearSVC__tol", 1e-4),
                random_state=seed
            )

        elif cls == "sklearn.linear_model.LogisticRegression":
            clf = LogisticRegression(
                C=config.get("LogisticRegression__C", 1.0),
                penalty=config.get("LogisticRegression__penalty", "l2"),
                dual=config.get("LogisticRegression__dual", False),
                max_iter=1000,
                random_state=seed
            )

        elif cls == "sklearn.linear_model.SGDClassifier":
            clf = SGDClassifier(
                loss=config.get("SGDClassifier__loss", "hinge"),
                alpha=config.get("SGDClassifier__alpha", 0.0001),
                learning_rate=config.get("SGDClassifier__learning_rate", "optimal"),
                fit_intercept=config.get("SGDClassifier__fit_intercept", True),
                l1_ratio=config.get("SGDClassifier__l1_ratio", 0.15),
                eta0=config.get("SGDClassifier__eta0", 0.01),
                power_t=config.get("SGDClassifier__power_t", 0.5),
                random_state=seed
            )

        elif cls == "sklearn.neural_network.MLPClassifier":
            clf = MLPClassifier(
                alpha=config.get("MLPClassifier__alpha", 0.0001),
                learning_rate_init=config.get("MLPClassifier__learning_rate_init", 0.001),
                max_iter=200,
                random_state=seed
            )

        elif cls == "xgboost.XGBClassifier":
            clf = XGBClassifier(
                n_estimators=100,
                max_depth=config.get("XGBClassifier__max_depth", 6),
                learning_rate=config.get("XGBClassifier__learning_rate", 0.3),
                subsample=config.get("XGBClassifier__subsample", 1.0),
                min_child_weight=config.get("XGBClassifier__min_child_weight", 1),
                verbosity=0,
                n_jobs=1,
                random_state=seed,
                use_label_encoder=False,
                eval_metric="logloss"
            )
        else:
            raise ValueError(cls)

        # ==================================================
        # Pipeline + CV
        # ==================================================
        steps = []
        if preprocessing is not None:
            steps.append(("preprocessing", preprocessing))
        if fe is not None:
            steps.append(("feature_engineering", fe))
        steps.append(("classifier", clf))

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