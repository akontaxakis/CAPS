import argparse, time, json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB
# === NEW ===
import socket

from AutoML_data_manager.data_manager import DataManager
from example_config import get_tpot_configspace_classifiers_for_SMAC4AC, \
    get_tpot_configspace_classifiers_for_SMAC4AC_full

#parser = argparse.ArgumentParser(description="Run Hyper with TPOT configuration.")
#parser.add_argument('dataset_id', type=str, help='The dataset ID argument')
#parser.add_argument('id', type=int, help='The experiment ID / random seed argument')
# === NEW ===
#args = parser.parse_args()
#data_id = args.dataset_id
#seed = args.id

data_id = "jannis"
seed = 7777
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def _cv_mean_score(pipe, Xb, yb, scoring, cv):
    from sklearn.model_selection import cross_val_score
    import numpy as np
    scores = cross_val_score(
        pipe, Xb, yb,
        scoring=scoring,
        cv=cv,
        n_jobs=1,
        error_score='raise'
    )
    return float(np.mean(scores))


from concurrent.futures import ProcessPoolExecutor, TimeoutError
import numpy as np
import time


def train(
    config,
    X,
    y,
    budget: float = 1.0,
    seed: int = 0,
    scoring: str = "accuracy",
    cv_splits: int = 2
):
    # ==================================================
    # Imports
    # ==================================================
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold

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

    # ==================================================
    # Preprocessing
    # ==================================================
    prep_name = config["preprocessing"]
    if prep_name == "None":
        preprocessing = None
    elif prep_name == "sklearn.preprocessing.Binarizer":
        preprocessing = Binarizer(
            threshold=config.get("Binarizer__threshold", 0.0)
        )
    elif prep_name == "sklearn.preprocessing.MaxAbsScaler":
        preprocessing = MaxAbsScaler()
    elif prep_name == "sklearn.preprocessing.MinMaxScaler":
        preprocessing = MinMaxScaler()
    elif prep_name == "sklearn.preprocessing.Normalizer":
        preprocessing = Normalizer(
            norm=config.get("Normalizer__norm", "l2")
        )
    elif prep_name == "sklearn.preprocessing.RobustScaler":
        preprocessing = RobustScaler()
    elif prep_name == "sklearn.preprocessing.StandardScaler":
        preprocessing = StandardScaler()
    else:
        raise ValueError(prep_name)

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
        # sklearn version safe
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
    # Classifier
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
    # Pipeline
    # ==================================================
    steps = []
    if preprocessing is not None:
        steps.append(("preprocessing", preprocessing))
    if fe is not None:
        steps.append(("feature_engineering", fe))
    steps.append(("classifier", clf))
    pipe = Pipeline(steps)

    # ==================================================
    # Budget subsampling
    # ==================================================
    frac = float(np.clip(budget, 0.05, 1.0))
    n = len(y)
    rng = np.random.RandomState(seed)
    m = max(50, int(frac * n))
    idx = rng.permutation(n)[:m]
    Xb, yb = X[idx], y[idx]

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    t0 = time.time()

    # ==================================================
    # Timed execution
    # ==================================================
    with ProcessPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_cv_mean_score, pipe, Xb, yb, scoring, cv)
        try:
            perf = fut.result(timeout=300)
            return 1.0 - perf, {
                "perf": perf,
                "elapsed_sec": time.time() - t0,
                "budget_frac": frac,
                "subsample_n": int(m),
            }
        except TimeoutError:
            fut.cancel()
            return 1.0, {
                "perf": 0.0,
                "elapsed_sec": time.time() - t0,
                "budget_frac": frac,
                "subsample_n": int(m),
                "exception": "TimeoutError",
            }
        except Exception as e:
            return 1.0, {
                "perf": 0.0,
                "elapsed_sec": time.time() - t0,
                "budget_frac": frac,
                "subsample_n": int(m),
                "exception": type(e).__name__,
                "msg": str(e),
            }


# ----- HpBandSter worker -----
class SKWorker(Worker):
    def __init__(self, X, y, **kwargs):
        super().__init__(**kwargs); self.X, self.y = X, y
    def compute(self, config, budget, **kwargs):
        loss, info = train(config, self.X, self.y, budget=budget, seed=seed, scoring='balanced_accuracy', cv_splits=2)
        return {'loss': loss, 'info': info}

def main():
    import Pyro4
    Pyro4.config.SERIALIZER = 'pickle'
    try:
        Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')
    except AttributeError:
        Pyro4.config.SERIALIZERS_ACCEPTED = set(getattr(Pyro4.config, 'SERIALIZERS_ACCEPTED', [])) | {'pickle'}

    # DataManager setup
    iris = DataManager(data_id, r'datasets', replace_missing=True, verbose=3)
    X = iris.data['X_train']
    y = iris.data['Y_train']

    cs = get_tpot_configspace_classifiers_for_SMAC4AC_full()
    exp_id = f"{seed}_{data_id}"
    run_id = f"hb_run_{exp_id}"

    # === CHANGED: bind to node IP and let OS pick a free port ===
    host_ip = socket.gethostbyname(socket.gethostname())
    NS = hpns.NameServer(run_id=run_id, host=host_ip, port=0)
    ns_host, ns_port = NS.start()

    # === CHANGED: point worker to this NS and host ===
    w = SKWorker(X, y, host=host_ip, run_id=run_id, nameserver=ns_host, nameserver_port=ns_port)
    w.run(background=True)

    from hpbandster.core.result import json_result_logger

    class NumpyJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return super().default(obj)

    json._default_encoder = NumpyJSONEncoder()

    # === CHANGED: pass nameserver + port and keep run_id consistent ===
    HB = BOHB(configspace=cs,
                   run_id=run_id,
                   nameserver=ns_host,
                   nameserver_port=ns_port,
                   min_budget=0.1,
                   max_budget=1.0,
                   eta=3,
                   result_logger=json_result_logger(directory='BOHB_'+exp_id, overwrite=True),
                   ping_interval=60)

    try:
        res = HB.run(n_iterations=999999)
    finally:
        HB.shutdown(shutdown_workers=True)
        NS.shutdown()


if __name__ == '__main__':
    main()
