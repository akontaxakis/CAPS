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
from example_config import get_tpot_configspace_classifiers_for_SMAC4AC

parser = argparse.ArgumentParser(description="Run Hyper with TPOT configuration.")
parser.add_argument('dataset_id', type=str, help='The dataset ID argument')
parser.add_argument('id', type=int, help='The experiment ID / random seed argument')
# === NEW ===
args = parser.parse_args()
data_id = args.dataset_id
seed = args.id




def _cv_mean_score(pipe, Xb, yb, scoring, cv):
    from sklearn.model_selection import cross_val_score
    import numpy as np
    scores = cross_val_score(pipe, Xb, yb, scoring=scoring, cv=cv, n_jobs=1, error_score='raise')
    return float(np.mean(scores))

from concurrent.futures import ProcessPoolExecutor, TimeoutError
# ----- budget-aware train (fixed n_estimators) -----
def train(config, X, y, budget: float = 1.0, seed: int = seed,
          scoring: str = 'balanced_accuracy', cv_splits: int = 2):
    pre_name = config['preprocessor']
    if pre_name == 'sklearn.preprocessing.Binarizer':
        pre = Binarizer(threshold=config.get('Binarizer__threshold', 0.0))
    elif pre_name == 'sklearn.preprocessing.MaxAbsScaler':
        pre = MaxAbsScaler()
    elif pre_name == 'sklearn.preprocessing.MinMaxScaler':
        pre = MinMaxScaler()
    elif pre_name == 'sklearn.preprocessing.Normalizer':
        pre = Normalizer(norm=config.get('Normalizer__norm', 'l2'))
    elif pre_name == 'sklearn.preprocessing.StandardScaler':
        pre = StandardScaler()
    elif pre_name == 'sklearn.decomposition.PCA':
        pre = PCA(svd_solver=config.get('PCA__svd_solver', 'randomized'),
                  iterated_power=config.get('PCA__iterated_power', 1))
    else:
        raise ValueError(f"Unknown preprocessor: {pre_name}")

    cls = config['classifier']
    if cls == 'sklearn.ensemble.RandomForestClassifier':
        clf = RandomForestClassifier(
            n_estimators=100,
            max_features=config['RandomForestClassifier__max_features'],
            min_samples_split=config['RandomForestClassifier__min_samples_split'],
            min_samples_leaf=config['RandomForestClassifier__min_samples_leaf'],
            bootstrap=config['RandomForestClassifier__bootstrap'],
            criterion=config['RandomForestClassifier__criterion'],
            random_state=seed
        )
    elif cls == 'sklearn.ensemble.GradientBoostingClassifier':
        clf = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=config['GradientBoostingClassifier__learning_rate'],
            max_depth=config['GradientBoostingClassifier__max_depth'],
            min_samples_split=config.get('GradientBoostingClassifier__min_samples_split', 2),
            min_samples_leaf=config.get('GradientBoostingClassifier__min_samples_leaf', 1),
            subsample=config.get('GradientBoostingClassifier__subsample', 1.0),
            max_features=config.get('GradientBoostingClassifier__max_features', None),
            random_state=seed
        )
    elif cls == 'sklearn.ensemble.ExtraTreesClassifier':
        clf = ExtraTreesClassifier(
            n_estimators=100,
            max_features=config['ExtraTreesClassifier__max_features'],
            min_samples_split=config['ExtraTreesClassifier__min_samples_split'],
            min_samples_leaf=config['ExtraTreesClassifier__min_samples_leaf'],
            criterion=config['ExtraTreesClassifier__criterion'],
            bootstrap=config['ExtraTreesClassifier__bootstrap'],
            random_state=seed
        )
    elif cls == 'sklearn.tree.DecisionTreeClassifier':
        clf = DecisionTreeClassifier(
            criterion=config['DecisionTreeClassifier__criterion'],
            max_depth=config['DecisionTreeClassifier__max_depth'],
            min_samples_split=config.get('DecisionTreeClassifier__min_samples_split', 2),
            min_samples_leaf=config.get('DecisionTreeClassifier__min_samples_leaf', 1),
            random_state=seed
        )
    elif cls == 'sklearn.neighbors.KNeighborsClassifier':
        clf = KNeighborsClassifier(
            n_neighbors=config['KNeighborsClassifier__n_neighbors'],
            weights=config['KNeighborsClassifier__weights'],
            p=config['KNeighborsClassifier__p']
        )
    elif cls == 'sklearn.naive_bayes.GaussianNB':
        clf = GaussianNB()
    elif cls == 'sklearn.naive_bayes.BernoulliNB':
        clf = BernoulliNB(
            alpha=config['BernoulliNB__alpha'],
            fit_prior=config['BernoulliNB__fit_prior']
        )
    else:
        raise ValueError(f"Unknown classifier: {cls}")

    pipe = Pipeline([('pre', pre), ('clf', clf)])

    # subsample by budget
    frac = float(np.clip(budget, 0.05, 1.0))
    n = len(y)
    rng = np.random.RandomState(seed)
    m = max(50, int(frac * n))
    idx = rng.permutation(n)[:m]
    Xb, yb = X[idx], y[idx]

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_cv_mean_score, pipe, Xb, yb, scoring, cv)
        try:
            perf = fut.result(timeout=300)  # timelimit is in SECONDS
            return 1.0 - perf, {
                'perf': perf,
                'elapsed_sec': time.time() - t0,
                'budget_frac': frac,
                'subsample_n': int(m)
            }
        except TimeoutError:
            # kill worker and report timeout as requested
            fut.cancel()
            return 1.0, {
                'perf': 0.0,  # score 0 on timeout
                'elapsed_sec': time.time() - t0,  # fixed at 300 on timeout
                'budget_frac': frac,
                'subsample_n': int(m),
                'exception': 'TimeoutError'
            }
        except Exception as e:
            return 1.0, {
                'perf': 0.0,
                'elapsed_sec': time.time() - t0,
                'budget_frac': frac,
                'subsample_n': int(m),
                'exception': type(e).__name__,
                'msg': str(e)
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

    cs = get_tpot_configspace_classifiers_for_SMAC4AC()
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
