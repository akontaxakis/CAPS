import yaml
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, EqualsCondition, Configuration


def get_tpot_configspace_classifiers_for_SMAC4AC():
    cs = ConfigurationSpace()

    # Define the main classifier hyperparameter with all classifier choices
    classifier = CategoricalHyperparameter(
        'classifier', [
            'sklearn.naive_bayes.GaussianNB',
            'sklearn.naive_bayes.BernoulliNB',
            'sklearn.tree.DecisionTreeClassifier',
            'sklearn.neighbors.KNeighborsClassifier',
            'sklearn.ensemble.ExtraTreesClassifier',
            'sklearn.ensemble.RandomForestClassifier',
            'sklearn.ensemble.GradientBoostingClassifier'
        ]
    )
    cs.add(classifier)

    # Define the main preprocessor hyperparameter with all preprocessor choices
    preprocessor = CategoricalHyperparameter(
        'preprocessor', [
            'sklearn.preprocessing.Binarizer',
            'sklearn.preprocessing.MaxAbsScaler',
            'sklearn.preprocessing.MinMaxScaler',
            'sklearn.preprocessing.Normalizer',
            'sklearn.decomposition.PCA',
            'sklearn.preprocessing.StandardScaler'
        ]
    )
    cs.add(preprocessor)

    # BernoulliNB
    bernoulli_nb_alpha = UniformFloatHyperparameter('BernoulliNB__alpha', lower=1e-3, upper=100, log=True)
    bernoulli_nb_fit_prior = CategoricalHyperparameter('BernoulliNB__fit_prior', [True, False])
    cs.add([bernoulli_nb_alpha, bernoulli_nb_fit_prior])
    cs.add(EqualsCondition(bernoulli_nb_alpha, classifier, 'sklearn.naive_bayes.BernoulliNB'))
    cs.add(EqualsCondition(bernoulli_nb_fit_prior, classifier, 'sklearn.naive_bayes.BernoulliNB'))

    # DecisionTreeClassifier
    decision_tree_criterion = CategoricalHyperparameter('DecisionTreeClassifier__criterion', ['gini', 'entropy'])
    decision_tree_max_depth = UniformIntegerHyperparameter('DecisionTreeClassifier__max_depth', lower=1, upper=10)
    decision_tree_min_samples_split = UniformIntegerHyperparameter('DecisionTreeClassifier__min_samples_split', lower=2, upper=20)
    decision_tree_min_samples_leaf = UniformIntegerHyperparameter('DecisionTreeClassifier__min_samples_leaf', lower=1, upper=20)
    cs.add([
        decision_tree_criterion,
        decision_tree_max_depth,
        decision_tree_min_samples_split,
        decision_tree_min_samples_leaf
    ])
    cs.add(EqualsCondition(decision_tree_criterion, classifier, 'sklearn.tree.DecisionTreeClassifier'))
    cs.add(EqualsCondition(decision_tree_max_depth, classifier, 'sklearn.tree.DecisionTreeClassifier'))
    cs.add(EqualsCondition(decision_tree_min_samples_split, classifier, 'sklearn.tree.DecisionTreeClassifier'))
    cs.add(EqualsCondition(decision_tree_min_samples_leaf, classifier, 'sklearn.tree.DecisionTreeClassifier'))

    # KNeighborsClassifier
    knn_n_neighbors = UniformIntegerHyperparameter('KNeighborsClassifier__n_neighbors', lower=1, upper=100)
    knn_weights = CategoricalHyperparameter('KNeighborsClassifier__weights', ['uniform', 'distance'])
    knn_p = CategoricalHyperparameter('KNeighborsClassifier__p', [1, 2])
    cs.add([knn_n_neighbors, knn_weights, knn_p])
    cs.add(EqualsCondition(knn_n_neighbors, classifier, 'sklearn.neighbors.KNeighborsClassifier'))
    cs.add(EqualsCondition(knn_weights, classifier, 'sklearn.neighbors.KNeighborsClassifier'))
    cs.add(EqualsCondition(knn_p, classifier, 'sklearn.neighbors.KNeighborsClassifier'))

    # ExtraTreesClassifier
    extra_trees_max_features = UniformFloatHyperparameter('ExtraTreesClassifier__max_features', lower=0.05, upper=1.0)
    extra_trees_min_samples_split = UniformIntegerHyperparameter('ExtraTreesClassifier__min_samples_split', lower=2, upper=20)
    extra_trees_min_samples_leaf = UniformIntegerHyperparameter('ExtraTreesClassifier__min_samples_leaf', lower=1, upper=20)
    extra_trees_criterion = CategoricalHyperparameter('ExtraTreesClassifier__criterion', ['gini', 'entropy'])
    extra_trees_bootstrap = CategoricalHyperparameter('ExtraTreesClassifier__bootstrap', [True, False])
    cs.add([
        extra_trees_max_features,
        extra_trees_min_samples_split,
        extra_trees_min_samples_leaf,
        extra_trees_criterion,
        extra_trees_bootstrap
    ])
    cs.add(EqualsCondition(extra_trees_max_features, classifier, 'sklearn.ensemble.ExtraTreesClassifier'))
    cs.add(EqualsCondition(extra_trees_min_samples_split, classifier, 'sklearn.ensemble.ExtraTreesClassifier'))
    cs.add(EqualsCondition(extra_trees_min_samples_leaf, classifier, 'sklearn.ensemble.ExtraTreesClassifier'))
    cs.add(EqualsCondition(extra_trees_criterion, classifier, 'sklearn.ensemble.ExtraTreesClassifier'))
    cs.add(EqualsCondition(extra_trees_bootstrap, classifier, 'sklearn.ensemble.ExtraTreesClassifier'))

    # RandomForestClassifier
    random_forest_max_features = UniformFloatHyperparameter('RandomForestClassifier__max_features', lower=0.05, upper=1.0)
    random_forest_min_samples_split = UniformIntegerHyperparameter('RandomForestClassifier__min_samples_split', lower=2, upper=20)
    random_forest_min_samples_leaf = UniformIntegerHyperparameter('RandomForestClassifier__min_samples_leaf', lower=1, upper=20)
    random_forest_criterion = CategoricalHyperparameter('RandomForestClassifier__criterion', ['gini', 'entropy'])
    random_forest_bootstrap = CategoricalHyperparameter('RandomForestClassifier__bootstrap', [True, False])
    cs.add([
        random_forest_max_features,
        random_forest_min_samples_split,
        random_forest_min_samples_leaf,
        random_forest_criterion,
        random_forest_bootstrap
    ])
    cs.add(EqualsCondition(random_forest_max_features, classifier, 'sklearn.ensemble.RandomForestClassifier'))
    cs.add(EqualsCondition(random_forest_min_samples_split, classifier, 'sklearn.ensemble.RandomForestClassifier'))
    cs.add(EqualsCondition(random_forest_min_samples_leaf, classifier, 'sklearn.ensemble.RandomForestClassifier'))
    cs.add(EqualsCondition(random_forest_criterion, classifier, 'sklearn.ensemble.RandomForestClassifier'))
    cs.add(EqualsCondition(random_forest_bootstrap, classifier, 'sklearn.ensemble.RandomForestClassifier'))

    # GradientBoostingClassifier
    gradient_boosting_learning_rate = UniformFloatHyperparameter('GradientBoostingClassifier__learning_rate', lower=1e-3, upper=1.0, log=True)
    gradient_boosting_max_depth = UniformIntegerHyperparameter('GradientBoostingClassifier__max_depth', lower=1, upper=10)
    gradient_boosting_min_samples_split = UniformIntegerHyperparameter('GradientBoostingClassifier__min_samples_split', lower=2, upper=20)
    gradient_boosting_min_samples_leaf = UniformIntegerHyperparameter('GradientBoostingClassifier__min_samples_leaf', lower=1, upper=20)
    gradient_boosting_subsample = UniformFloatHyperparameter('GradientBoostingClassifier__subsample', lower=0.05, upper=1.0)
    gradient_boosting_max_features = UniformFloatHyperparameter('GradientBoostingClassifier__max_features', lower=0.05, upper=1.0)
    cs.add([
        gradient_boosting_learning_rate,
        gradient_boosting_max_depth,
        gradient_boosting_min_samples_split,
        gradient_boosting_min_samples_leaf,
        gradient_boosting_subsample,
        gradient_boosting_max_features
    ])
    cs.add(EqualsCondition(gradient_boosting_learning_rate, classifier, 'sklearn.ensemble.GradientBoostingClassifier'))
    cs.add(EqualsCondition(gradient_boosting_max_depth, classifier, 'sklearn.ensemble.GradientBoostingClassifier'))
    cs.add(EqualsCondition(gradient_boosting_min_samples_split, classifier, 'sklearn.ensemble.GradientBoostingClassifier'))
    cs.add(EqualsCondition(gradient_boosting_min_samples_leaf, classifier, 'sklearn.ensemble.GradientBoostingClassifier'))
    cs.add(EqualsCondition(gradient_boosting_subsample, classifier, 'sklearn.ensemble.GradientBoostingClassifier'))
    cs.add(EqualsCondition(gradient_boosting_max_features, classifier, 'sklearn.ensemble.GradientBoostingClassifier'))

    # Define Preprocessing Hyperparameters
    binarizer_threshold = UniformFloatHyperparameter('Binarizer__threshold', lower=0.0, upper=1.0)
    cs.add(binarizer_threshold)
    cs.add(EqualsCondition(binarizer_threshold, preprocessor, 'sklearn.preprocessing.Binarizer'))

    pca_svd_solver = CategoricalHyperparameter('PCA__svd_solver', ['randomized'])
    pca_iterated_power = UniformIntegerHyperparameter('PCA__iterated_power', lower=1, upper=10)
    cs.add([pca_svd_solver, pca_iterated_power])
    cs.add(EqualsCondition(pca_svd_solver, preprocessor, 'sklearn.decomposition.PCA'))
    cs.add(EqualsCondition(pca_iterated_power, preprocessor, 'sklearn.decomposition.PCA'))

    normalizer_norm = CategoricalHyperparameter('Normalizer__norm', ['l1', 'l2', 'max'])
    cs.add(normalizer_norm)
    cs.add(EqualsCondition(normalizer_norm, preprocessor, 'sklearn.preprocessing.Normalizer'))

    return cs



from ConfigSpace import (
    ConfigurationSpace,
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    EqualsCondition,
)


from ConfigSpace import (
    ConfigurationSpace,
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from ConfigSpace.conditions import EqualsCondition


def get_tpot_configspace_classifiers_for_SMAC4AC_full():
    cs = ConfigurationSpace()

    # ==================================================
    # Pipeline structure
    # ==================================================

    preprocessing = CategoricalHyperparameter(
        "preprocessing",
        [
            "None",
            "sklearn.preprocessing.Binarizer",
            "sklearn.preprocessing.MaxAbsScaler",
            "sklearn.preprocessing.MinMaxScaler",
            "sklearn.preprocessing.Normalizer",
            "sklearn.preprocessing.RobustScaler",
            "sklearn.preprocessing.StandardScaler",
        ],
    )

    feature_engineering = CategoricalHyperparameter(
        "feature_engineering",
        [
            "None",
            "sklearn.decomposition.FastICA",
            "sklearn.cluster.FeatureAgglomeration",
            "sklearn.kernel_approximation.Nystroem",
            "sklearn.decomposition.PCA",
            "sklearn.preprocessing.PolynomialFeatures",
            "sklearn.kernel_approximation.RBFSampler",
            "tpot.builtins.ZeroCount",
            "sklearn.feature_selection.SelectFwe",
            "sklearn.feature_selection.SelectPercentile",
            "sklearn.feature_selection.VarianceThreshold",
            "sklearn.feature_selection.RFE",
            "sklearn.feature_selection.SelectFromModel",
        ],
    )

    classifier = CategoricalHyperparameter(
        "classifier",
        [
            "sklearn.naive_bayes.GaussianNB",
            "sklearn.naive_bayes.BernoulliNB",
            "sklearn.naive_bayes.MultinomialNB",
            "sklearn.tree.DecisionTreeClassifier",
            "sklearn.ensemble.ExtraTreesClassifier",
            "sklearn.ensemble.RandomForestClassifier",
            "sklearn.ensemble.GradientBoostingClassifier",
            "sklearn.neighbors.KNeighborsClassifier",
            "sklearn.svm.LinearSVC",
            "sklearn.linear_model.LogisticRegression",
            "sklearn.linear_model.SGDClassifier",
            "sklearn.neural_network.MLPClassifier",
            "xgboost.XGBClassifier",
        ],
    )

    cs.add([preprocessing, feature_engineering, classifier])

    # ==================================================
    # Preprocessing
    # ==================================================

    bin_thresh = UniformFloatHyperparameter("Binarizer__threshold", 0.0, 1.0)
    norm_norm = CategoricalHyperparameter("Normalizer__norm", ["l1", "l2", "max"])
    cs.add([bin_thresh, norm_norm])
    cs.add(EqualsCondition(bin_thresh, preprocessing, "sklearn.preprocessing.Binarizer"))
    cs.add(EqualsCondition(norm_norm, preprocessing, "sklearn.preprocessing.Normalizer"))

    # ==================================================
    # Feature engineering
    # ==================================================

    ica_tol = UniformFloatHyperparameter("FastICA__tol", 0.0, 1.0)
    cs.add(ica_tol)
    cs.add(EqualsCondition(ica_tol, feature_engineering, "sklearn.decomposition.FastICA"))

    agg_link = CategoricalHyperparameter(
        "FeatureAgglomeration__linkage", ["ward", "complete", "average"]
    )
    agg_aff = CategoricalHyperparameter(
        "FeatureAgglomeration__affinity",
        ["euclidean", "l1", "l2", "manhattan", "cosine"],
    )
    cs.add([agg_link, agg_aff])
    cs.add(EqualsCondition(agg_link, feature_engineering, "sklearn.cluster.FeatureAgglomeration"))
    cs.add(EqualsCondition(agg_aff, feature_engineering, "sklearn.cluster.FeatureAgglomeration"))

    nys_kernel = CategoricalHyperparameter(
        "Nystroem__kernel",
        ["rbf", "cosine", "chi2", "laplacian", "polynomial", "poly", "linear", "additive_chi2", "sigmoid"],
    )
    nys_gamma = UniformFloatHyperparameter("Nystroem__gamma", 0.0, 1.0)
    nys_comp = UniformIntegerHyperparameter("Nystroem__n_components", 1, 10)
    cs.add([nys_kernel, nys_gamma, nys_comp])
    for hp in [nys_kernel, nys_gamma, nys_comp]:
        cs.add(EqualsCondition(hp, feature_engineering, "sklearn.kernel_approximation.Nystroem"))

    pca_solver = CategoricalHyperparameter("PCA__svd_solver", ["randomized"])
    pca_power = UniformIntegerHyperparameter("PCA__iterated_power", 1, 10)
    cs.add([pca_solver, pca_power])
    for hp in [pca_solver, pca_power]:
        cs.add(EqualsCondition(hp, feature_engineering, "sklearn.decomposition.PCA"))

    poly_deg = CategoricalHyperparameter("PolynomialFeatures__degree", [2])
    poly_bias = CategoricalHyperparameter("PolynomialFeatures__include_bias", [False])
    poly_inter = CategoricalHyperparameter("PolynomialFeatures__interaction_only", [False])
    cs.add([poly_deg, poly_bias, poly_inter])
    for hp in [poly_deg, poly_bias, poly_inter]:
        cs.add(EqualsCondition(hp, feature_engineering, "sklearn.preprocessing.PolynomialFeatures"))

    rbf_gamma = UniformFloatHyperparameter("RBFSampler__gamma", 0.0, 1.0)
    cs.add(rbf_gamma)
    cs.add(EqualsCondition(rbf_gamma, feature_engineering, "sklearn.kernel_approximation.RBFSampler"))

    fwe_alpha = UniformFloatHyperparameter("SelectFwe__alpha", 0.0, 0.05)
    sp_pct = UniformIntegerHyperparameter("SelectPercentile__percentile", 1, 99)
    vt_thresh = CategoricalHyperparameter(
        "VarianceThreshold__threshold",
        [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
    )
    cs.add([fwe_alpha, sp_pct, vt_thresh])
    cs.add(EqualsCondition(fwe_alpha, feature_engineering, "sklearn.feature_selection.SelectFwe"))
    cs.add(EqualsCondition(sp_pct, feature_engineering, "sklearn.feature_selection.SelectPercentile"))
    cs.add(EqualsCondition(vt_thresh, feature_engineering, "sklearn.feature_selection.VarianceThreshold"))

    sfm_thresh = UniformFloatHyperparameter("SelectFromModel__threshold", 0.0, 1.0)
    cs.add(sfm_thresh)
    cs.add(EqualsCondition(sfm_thresh, feature_engineering, "sklearn.feature_selection.SelectFromModel"))

    rfe_step = UniformFloatHyperparameter("RFE__step", 0.05, 1.0)
    cs.add(rfe_step)
    cs.add(EqualsCondition(rfe_step, feature_engineering, "sklearn.feature_selection.RFE"))

    # ==================================================
    # Classifiers
    # ==================================================

    # Bernoulli / Multinomial NB
    bnb_alpha = CategoricalHyperparameter("BernoulliNB__alpha", [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
    bnb_fit = CategoricalHyperparameter("BernoulliNB__fit_prior", [True, False])
    mnb_alpha = CategoricalHyperparameter("MultinomialNB__alpha", [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
    mnb_fit = CategoricalHyperparameter("MultinomialNB__fit_prior", [True, False])
    cs.add([bnb_alpha, bnb_fit, mnb_alpha, mnb_fit])
    for hp in [bnb_alpha, bnb_fit]:
        cs.add(EqualsCondition(hp, classifier, "sklearn.naive_bayes.BernoulliNB"))
    for hp in [mnb_alpha, mnb_fit]:
        cs.add(EqualsCondition(hp, classifier, "sklearn.naive_bayes.MultinomialNB"))

    # KNN
    knn_k = UniformIntegerHyperparameter("KNeighborsClassifier__n_neighbors", 1, 100)
    knn_w = CategoricalHyperparameter("KNeighborsClassifier__weights", ["uniform", "distance"])
    knn_p = CategoricalHyperparameter("KNeighborsClassifier__p", [1, 2])
    cs.add([knn_k, knn_w, knn_p])
    for hp in [knn_k, knn_w, knn_p]:
        cs.add(EqualsCondition(hp, classifier, "sklearn.neighbors.KNeighborsClassifier"))

    # DecisionTree
    dt_crit = CategoricalHyperparameter("DecisionTreeClassifier__criterion", ["gini", "entropy"])
    dt_depth = CategoricalHyperparameter("DecisionTreeClassifier__max_depth", [None, 5, 10, 20, 50])
    dt_mss = UniformIntegerHyperparameter("DecisionTreeClassifier__min_samples_split", 2, 20)
    dt_msl = UniformIntegerHyperparameter("DecisionTreeClassifier__min_samples_leaf", 1, 20)
    cs.add([dt_crit, dt_depth, dt_mss, dt_msl])
    for hp in [dt_crit, dt_depth, dt_mss, dt_msl]:
        cs.add(EqualsCondition(hp, classifier, "sklearn.tree.DecisionTreeClassifier"))

    # RandomForest
    rf_feat = CategoricalHyperparameter("RandomForestClassifier__max_features", ["sqrt", "log2", None])
    rf_mss = UniformIntegerHyperparameter("RandomForestClassifier__min_samples_split", 2, 20)
    rf_msl = UniformIntegerHyperparameter("RandomForestClassifier__min_samples_leaf", 1, 20)
    rf_crit = CategoricalHyperparameter("RandomForestClassifier__criterion", ["gini", "entropy"])
    rf_boot = CategoricalHyperparameter("RandomForestClassifier__bootstrap", [True, False])
    cs.add([rf_feat, rf_mss, rf_msl, rf_crit, rf_boot])
    for hp in [rf_feat, rf_mss, rf_msl, rf_crit, rf_boot]:
        cs.add(EqualsCondition(hp, classifier, "sklearn.ensemble.RandomForestClassifier"))

    # ExtraTrees
    et_feat = CategoricalHyperparameter("ExtraTreesClassifier__max_features", ["sqrt", "log2", None])
    et_mss = UniformIntegerHyperparameter("ExtraTreesClassifier__min_samples_split", 2, 20)
    et_msl = UniformIntegerHyperparameter("ExtraTreesClassifier__min_samples_leaf", 1, 20)
    et_crit = CategoricalHyperparameter("ExtraTreesClassifier__criterion", ["gini", "entropy"])
    et_boot = CategoricalHyperparameter("ExtraTreesClassifier__bootstrap", [True, False])
    cs.add([et_feat, et_mss, et_msl, et_crit, et_boot])
    for hp in [et_feat, et_mss, et_msl, et_crit, et_boot]:
        cs.add(EqualsCondition(hp, classifier, "sklearn.ensemble.ExtraTreesClassifier"))

    # GradientBoosting
    gb_lr = UniformFloatHyperparameter("GradientBoostingClassifier__learning_rate", 0.01, 1.0, log=True)
    gb_depth = UniformIntegerHyperparameter("GradientBoostingClassifier__max_depth", 1, 5)
    gb_mss = UniformIntegerHyperparameter("GradientBoostingClassifier__min_samples_split", 2, 20)
    gb_msl = UniformIntegerHyperparameter("GradientBoostingClassifier__min_samples_leaf", 1, 20)
    gb_sub = UniformFloatHyperparameter("GradientBoostingClassifier__subsample", 0.5, 1.0)
    gb_feat = CategoricalHyperparameter("GradientBoostingClassifier__max_features", ["sqrt", "log2", None])
    cs.add([gb_lr, gb_depth, gb_mss, gb_msl, gb_sub, gb_feat])
    for hp in [gb_lr, gb_depth, gb_mss, gb_msl, gb_sub, gb_feat]:
        cs.add(EqualsCondition(hp, classifier, "sklearn.ensemble.GradientBoostingClassifier"))

    # Linear models / NN
    sgd_loss = CategoricalHyperparameter("SGDClassifier__loss", ["hinge", "log_loss"])
    sgd_alpha = UniformFloatHyperparameter("SGDClassifier__alpha", 1e-6, 1e-1, log=True)
    sgd_lr = CategoricalHyperparameter("SGDClassifier__learning_rate", ["optimal", "invscaling", "adaptive"])
    sgd_fit = CategoricalHyperparameter("SGDClassifier__fit_intercept", [True, False])
    sgd_l1 = UniformFloatHyperparameter("SGDClassifier__l1_ratio", 0.0, 1.0)
    sgd_eta = UniformFloatHyperparameter("SGDClassifier__eta0", 1e-4, 1.0, log=True)
    sgd_pow = UniformFloatHyperparameter("SGDClassifier__power_t", 0.1, 0.9)
    cs.add([sgd_loss, sgd_alpha, sgd_lr, sgd_fit, sgd_l1, sgd_eta, sgd_pow])
    for hp in [sgd_loss, sgd_alpha, sgd_lr, sgd_fit, sgd_l1, sgd_eta, sgd_pow]:
        cs.add(EqualsCondition(hp, classifier, "sklearn.linear_model.SGDClassifier"))

    mlp_alpha = UniformFloatHyperparameter("MLPClassifier__alpha", 1e-6, 1e-1, log=True)
    mlp_lr = UniformFloatHyperparameter("MLPClassifier__learning_rate_init", 1e-4, 1e-1, log=True)
    cs.add([mlp_alpha, mlp_lr])
    cs.add(EqualsCondition(mlp_alpha, classifier, "sklearn.neural_network.MLPClassifier"))
    cs.add(EqualsCondition(mlp_lr, classifier, "sklearn.neural_network.MLPClassifier"))

    # XGBoost
    xgb_depth = UniformIntegerHyperparameter("XGBClassifier__max_depth", 1, 10)
    xgb_lr = UniformFloatHyperparameter("XGBClassifier__learning_rate", 0.01, 1.0, log=True)
    xgb_sub = UniformFloatHyperparameter("XGBClassifier__subsample", 0.5, 1.0)
    xgb_child = UniformIntegerHyperparameter("XGBClassifier__min_child_weight", 1, 10)
    cs.add([xgb_depth, xgb_lr, xgb_sub, xgb_child])
    for hp in [xgb_depth, xgb_lr, xgb_sub, xgb_child]:
        cs.add(EqualsCondition(hp, classifier, "xgboost.XGBClassifier"))

    return cs
def update_log_filename(file_path: str, new_filename: str) -> str:
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Update the filename in the file handler
    if 'handlers' in config and 'file' in config['handlers']:
        config['handlers']['file']['filename'] = new_filename

    # Define the output file path with 'log_' prefix and the new filename
    output_file_path = f"log_{new_filename}"

    # Save the modified configuration to the new file
    with open(output_file_path, 'w') as file:
        yaml.safe_dump(config, file)

    print(f"Log filename updated to '{new_filename}' and saved as '{output_file_path}'.")
    return output_file_path

def adapt_meta_config_to_full_cs(meta_cfg: dict, cs) -> dict:
    cfg = meta_cfg.copy()

    # Extract legacy preprocessor
    legacy = cfg.pop("preprocessor", None)

    # Default structure
    cfg.setdefault("preprocessing", "None")
    cfg.setdefault("feature_engineering", "None")

    # Map legacy preprocessor correctly
    if legacy is not None:
        if legacy.startswith("sklearn.preprocessing."):
            cfg["preprocessing"] = legacy
        else:
            # PCA, FastICA, etc. belong to feature_engineering
            cfg["feature_engineering"] = legacy

    # Remove hyperparameters not in the ConfigSpace
    valid = set(cs.get_hyperparameter_names())
    cfg = {k: v for k, v in cfg.items() if k in valid}

    return cfg


def get_metalearning_pipelines(data_id):
    configspace = get_tpot_configspace_classifiers_for_SMAC4AC()
    if data_id == "digits":
        INITIAL_CONFIGS_VALUES = [
            # id 2
            {
                "classifier": "sklearn.neighbors.KNeighborsClassifier",
                "preprocessor": "sklearn.decomposition.PCA",
                "KNeighborsClassifier__n_neighbors": 4,
                "KNeighborsClassifier__p": 2,
                "KNeighborsClassifier__weights": "distance",
                # optional PCA params if desired:
                # "PCA__svd_solver": "randomized",
                # "PCA__iterated_power": 1,
            },
            # id 3
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.04634380160611007,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 41 clipped to 20
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
            # id 4
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.09546265146045475,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 2,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
            # id 5
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.09750328007832798,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 25 -> 20
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
            # id 6
            {
                "classifier": "sklearn.ensemble.ExtraTreesClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "ExtraTreesClassifier__bootstrap": True,
                "ExtraTreesClassifier__criterion": "gini",
                "ExtraTreesClassifier__max_features": 0.37705188916038523,
                "ExtraTreesClassifier__min_samples_leaf": 2,
                "ExtraTreesClassifier__min_samples_split": 3,
            },
            # id 7
            {
                "classifier": "sklearn.naive_bayes.GaussianNB",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
            },
            # id 8
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.05972079854295879,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 2,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
            # id 9
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.16262682406125173,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
            # id 10
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.926283631486858,
                "RandomForestClassifier__min_samples_leaf": 7,
                "RandomForestClassifier__min_samples_split": 2,
            },
            # id 11
            {
                "classifier": "sklearn.ensemble.ExtraTreesClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "ExtraTreesClassifier__bootstrap": False,
                "ExtraTreesClassifier__criterion": "entropy",
                "ExtraTreesClassifier__max_features": 0.6128603428070196,
                "ExtraTreesClassifier__min_samples_leaf": 1,
                "ExtraTreesClassifier__min_samples_split": 3,
            },
            # id 12
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.09884140378258977,
                "RandomForestClassifier__min_samples_leaf": 6,
                "RandomForestClassifier__min_samples_split": 13,
            },
            # id 13
            {
                "classifier": "sklearn.tree.DecisionTreeClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "DecisionTreeClassifier__criterion": "gini",
                "DecisionTreeClassifier__max_depth": 5,  # picked within [1..10]
                "DecisionTreeClassifier__min_samples_split": 20,
                "DecisionTreeClassifier__min_samples_leaf": 4,
            },
            # id 14
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.decomposition.PCA",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "gini",
                "RandomForestClassifier__max_features": 0.9331254454871041,
                "RandomForestClassifier__min_samples_leaf": 2,
                "RandomForestClassifier__min_samples_split": 20,
                # optional PCA params:
                # "PCA__svd_solver": "randomized",
                # "PCA__iterated_power": 1,
            },
            # id 15
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.11042628136263043,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 22 -> 20
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
            # id 16
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.41808321658160696,
                "RandomForestClassifier__min_samples_leaf": 1,
                "RandomForestClassifier__min_samples_split": 4,
            },
            # id 17
            {
                "classifier": "sklearn.neighbors.KNeighborsClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "KNeighborsClassifier__n_neighbors": 2,
                "KNeighborsClassifier__p": 2,
                "KNeighborsClassifier__weights": "uniform",
            },
            # id 18
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.7323115919225983,
                "RandomForestClassifier__min_samples_leaf": 15,
                "RandomForestClassifier__min_samples_split": 6,
            },
            # id 19
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.6779841015398226,
                "RandomForestClassifier__min_samples_leaf": 14,
                "RandomForestClassifier__min_samples_split": 14,
            },
            # id 20
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.7708651296941559,
                "RandomForestClassifier__min_samples_leaf": 13,
                "RandomForestClassifier__min_samples_split": 2,
            },
            # id 21
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.03162215674470446,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 1,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
        ]
    elif data_id == "dilbert":
        INITIAL_CONFIGS_VALUES = [
            # 1) GB (weighting, early_stop=train)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.04634380160611007,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 41 -> clip to 20
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
            # 2) KNN + PCA (weighting)
            {
                "classifier": "sklearn.neighbors.KNeighborsClassifier",
                "preprocessor": "sklearn.decomposition.PCA",
                "KNeighborsClassifier__n_neighbors": 4,
                "KNeighborsClassifier__p": 2,
                "KNeighborsClassifier__weights": "distance",
                "PCA__svd_solver": "randomized",
                "PCA__iterated_power": 6,
            },
            # 3) GB (weighting, early_stop=valid)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.09750328007832798,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 25 -> 20
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
            # 4) GB (weighting, early_stop=off)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.09546265146045475,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 2,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
            # 5) GaussianNB (weighting)
            {
                "classifier": "sklearn.naive_bayes.GaussianNB",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
            },
            # 6) GB (none, early_stop=train)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.05972079854295879,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 2,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
            # 7) ExtraTrees (none)
            {
                "classifier": "sklearn.ensemble.ExtraTreesClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "ExtraTreesClassifier__bootstrap": True,
                "ExtraTreesClassifier__criterion": "gini",
                "ExtraTreesClassifier__max_features": 0.37705188916038523,
                "ExtraTreesClassifier__min_samples_leaf": 2,
                "ExtraTreesClassifier__min_samples_split": 3,
            },
            # 8) RandomForest (none)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.926283631486858,
                "RandomForestClassifier__min_samples_leaf": 7,
                "RandomForestClassifier__min_samples_split": 2,
            },
            # 9) GB (none, early_stop=off, high LR)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.16262682406125173,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
            # 10) DecisionTree (weighting)
            {
                "classifier": "sklearn.tree.DecisionTreeClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "DecisionTreeClassifier__criterion": "gini",
                "DecisionTreeClassifier__max_depth": 5,
                "DecisionTreeClassifier__min_samples_split": 20,
                "DecisionTreeClassifier__min_samples_leaf": 4,
            },
            # 11) RandomForest + PCA (none)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.decomposition.PCA",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "gini",
                "RandomForestClassifier__max_features": 0.9331254454871041,
                "RandomForestClassifier__min_samples_leaf": 2,
                "RandomForestClassifier__min_samples_split": 20,
                "PCA__svd_solver": "randomized",
                "PCA__iterated_power": 6,
            },
            # 12) RandomForest (none)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.09884140378258977,
                "RandomForestClassifier__min_samples_leaf": 6,
                "RandomForestClassifier__min_samples_split": 13,
            },
            # 13) ExtraTrees (none)
            {
                "classifier": "sklearn.ensemble.ExtraTreesClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "ExtraTreesClassifier__bootstrap": False,
                "ExtraTreesClassifier__criterion": "entropy",
                "ExtraTreesClassifier__max_features": 0.6128603428070196,
                "ExtraTreesClassifier__min_samples_leaf": 1,
                "ExtraTreesClassifier__min_samples_split": 3,
            },
            # 14) GB (weighting, early_stop=off)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.11042628136263043,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 22 -> 20
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
            # 15) RandomForest (weighting)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.41808321658160696,
                "RandomForestClassifier__min_samples_leaf": 1,
                "RandomForestClassifier__min_samples_split": 4,
            },
            # 16) RandomForest (weighting)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.7323115919225983,
                "RandomForestClassifier__min_samples_leaf": 15,
                "RandomForestClassifier__min_samples_split": 6,
            },
            # 17) KNN (weighting)
            {
                "classifier": "sklearn.neighbors.KNeighborsClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "KNeighborsClassifier__n_neighbors": 2,
                "KNeighborsClassifier__p": 2,
                "KNeighborsClassifier__weights": "uniform",
            },
            # 18) RandomForest (weighting)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.6779841015398226,
                "RandomForestClassifier__min_samples_leaf": 14,
                "RandomForestClassifier__min_samples_split": 14,
            },
            # 19) RandomForest (none)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.7708651296941559,
                "RandomForestClassifier__min_samples_leaf": 13,
                "RandomForestClassifier__min_samples_split": 2,
            },
            # 20) GB (none, early_stop=off, low LR)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.03162215674470446,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 1,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
        ]
    elif data_id == "dionis":
        INITIAL_CONFIGS_VALUES = [
            # id 1
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "gini",
                "RandomForestClassifier__max_features": 0.5,
                "RandomForestClassifier__min_samples_leaf": 1,
                "RandomForestClassifier__min_samples_split": 2,
            },
            # id 2
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.03162215674470446,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 1,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
            # id 3
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.09546265146045475,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 2,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
            # id 4
            {
                "classifier": "sklearn.neighbors.KNeighborsClassifier",
                "preprocessor": "sklearn.decomposition.PCA",
                "KNeighborsClassifier__n_neighbors": 4,
                "KNeighborsClassifier__p": 2,
                "KNeighborsClassifier__weights": "distance",
                "PCA__svd_solver": "randomized",
                "PCA__iterated_power": 6,
            },
            # id 5
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.04634380160611007,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 41 → clip to 20
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
            # id 6
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.09750328007832798,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 25 → 20
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
            # id 7
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.11042628136263043,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 22 → 20
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
            # id 8
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.09884140378258977,
                "RandomForestClassifier__min_samples_leaf": 6,
                "RandomForestClassifier__min_samples_split": 13,
            },
            # id 9
            {
                "classifier": "sklearn.naive_bayes.GaussianNB",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
            },
            # id 10
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.05972079854295879,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 2,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
            # id 11
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.decomposition.PCA",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "gini",
                "RandomForestClassifier__max_features": 0.9331254454871041,
                "RandomForestClassifier__min_samples_leaf": 2,
                "RandomForestClassifier__min_samples_split": 20,
                "PCA__svd_solver": "randomized",
                "PCA__iterated_power": 6,
            },
            # id 12
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.16262682406125173,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
            # id 13
            {
                "classifier": "sklearn.tree.DecisionTreeClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "DecisionTreeClassifier__criterion": "gini",
                "DecisionTreeClassifier__max_depth": 5,
                "DecisionTreeClassifier__min_samples_split": 20,
                "DecisionTreeClassifier__min_samples_leaf": 4,
            },
            # id 14
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.926283631486858,
                "RandomForestClassifier__min_samples_leaf": 7,
                "RandomForestClassifier__min_samples_split": 2,
            },
            # id 15
            {
                "classifier": "sklearn.ensemble.ExtraTreesClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "ExtraTreesClassifier__bootstrap": False,
                "ExtraTreesClassifier__criterion": "entropy",
                "ExtraTreesClassifier__max_features": 0.6128603428070196,
                "ExtraTreesClassifier__min_samples_leaf": 1,
                "ExtraTreesClassifier__min_samples_split": 3,
            },
            # id 16
            {
                "classifier": "sklearn.ensemble.ExtraTreesClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "ExtraTreesClassifier__bootstrap": True,
                "ExtraTreesClassifier__criterion": "gini",
                "ExtraTreesClassifier__max_features": 0.37705188916038523,
                "ExtraTreesClassifier__min_samples_leaf": 2,
                "ExtraTreesClassifier__min_samples_split": 3,
            },
            # id 17
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.41808321658160696,
                "RandomForestClassifier__min_samples_leaf": 1,
                "RandomForestClassifier__min_samples_split": 4,
            },
            # id 18
            {
                "classifier": "sklearn.neighbors.KNeighborsClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "KNeighborsClassifier__n_neighbors": 2,
                "KNeighborsClassifier__p": 2,
                "KNeighborsClassifier__weights": "uniform",
            },
            # id 19
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.7323115919225983,
                "RandomForestClassifier__min_samples_leaf": 15,
                "RandomForestClassifier__min_samples_split": 6,
            },
            # id 20
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.6779841015398226,
                "RandomForestClassifier__min_samples_leaf": 14,
                "RandomForestClassifier__min_samples_split": 14,
            },
            # id 21
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.7708651296941559,
                "RandomForestClassifier__min_samples_leaf": 13,
                "RandomForestClassifier__min_samples_split": 2,
            },
        ]
    elif data_id == "fabert":
        INITIAL_CONFIGS_VALUES = [
            # 2  gradient_boosting (weighting, early_stop=train)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.04634380160611007,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 41 -> 20 (cap)
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 3  k_nearest_neighbors + PCA (weighting)
            {
                "classifier": "sklearn.neighbors.KNeighborsClassifier",
                "preprocessor": "sklearn.decomposition.PCA",
                "KNeighborsClassifier__n_neighbors": 4,
                "KNeighborsClassifier__p": 2,
                "KNeighborsClassifier__weights": "distance",
                # PCA defaults to satisfy your ConfigSpace
                "PCA__svd_solver": "randomized",
                "PCA__iterated_power": 6,
            },

            # 4  gradient_boosting (weighting, early_stop=valid)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.09750328007832798,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 25 -> 20 (cap)
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 5  gradient_boosting (weighting, early_stop=off)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.09546265146045475,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 2,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 6  extra_trees (none, bootstrap=True, gini)
            {
                "classifier": "sklearn.ensemble.ExtraTreesClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "ExtraTreesClassifier__bootstrap": True,
                "ExtraTreesClassifier__criterion": "gini",
                "ExtraTreesClassifier__max_features": 0.37705188916038523,
                "ExtraTreesClassifier__min_samples_leaf": 2,
                "ExtraTreesClassifier__min_samples_split": 3,
            },

            # 7  extra_trees (none, bootstrap=False, entropy)
            {
                "classifier": "sklearn.ensemble.ExtraTreesClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "ExtraTreesClassifier__bootstrap": False,
                "ExtraTreesClassifier__criterion": "entropy",
                "ExtraTreesClassifier__max_features": 0.6128603428070196,
                "ExtraTreesClassifier__min_samples_leaf": 1,
                "ExtraTreesClassifier__min_samples_split": 3,
            },

            # 8 gradient_boosting (none, early_stop=train, low depth)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.05972079854295879,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 2,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 9 gaussian_nb (weighting)
            {
                "classifier": "sklearn.naive_bayes.GaussianNB",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
            },

            # 10 random_forest (none, bootstrap=True, entropy, tiny max_features)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.09884140378258977,
                "RandomForestClassifier__min_samples_leaf": 6,
                "RandomForestClassifier__min_samples_split": 13,
            },

            # 11 random_forest (none, bootstrap=True, entropy, large max_features)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.926283631486858,
                "RandomForestClassifier__min_samples_leaf": 7,
                "RandomForestClassifier__min_samples_split": 2,
            },

            # 12 gradient_boosting (none, early_stop=off, high learning_rate)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.16262682406125173,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 13 random_forest + PCA (none, gini)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.decomposition.PCA",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "gini",
                "RandomForestClassifier__max_features": 0.9331254454871041,
                "RandomForestClassifier__min_samples_leaf": 2,
                "RandomForestClassifier__min_samples_split": 20,
                # PCA defaults to satisfy your ConfigSpace
                "PCA__svd_solver": "randomized",
                "PCA__iterated_power": 6,
            },

            # 14 decision_tree (weighting)
            {
                "classifier": "sklearn.tree.DecisionTreeClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "DecisionTreeClassifier__criterion": "gini",
                "DecisionTreeClassifier__max_depth": 5,  # derived from your factor
                "DecisionTreeClassifier__min_samples_split": 20,
                "DecisionTreeClassifier__min_samples_leaf": 4,
            },

            # 15 gradient_boosting (weighting, early_stop=off)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.11042628136263043,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 22 -> 20 (cap)
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 16 random_forest (weighting, bootstrap=False, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.41808321658160696,
                "RandomForestClassifier__min_samples_leaf": 1,
                "RandomForestClassifier__min_samples_split": 4,
            },

            # 17 k_nearest_neighbors (weighting)
            {
                "classifier": "sklearn.neighbors.KNeighborsClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "KNeighborsClassifier__n_neighbors": 2,
                "KNeighborsClassifier__p": 2,
                "KNeighborsClassifier__weights": "uniform",
            },

            # 18 random_forest (weighting, bootstrap=False, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.7323115919225983,
                "RandomForestClassifier__min_samples_leaf": 15,
                "RandomForestClassifier__min_samples_split": 6,
            },

            # 19 random_forest (weighting, bootstrap=True, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.6779841015398226,
                "RandomForestClassifier__min_samples_leaf": 14,
                "RandomForestClassifier__min_samples_split": 14,
            },

            # 20 random_forest (none, bootstrap=False, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.7708651296941559,
                "RandomForestClassifier__min_samples_leaf": 13,
                "RandomForestClassifier__min_samples_split": 2,
            },

            # 21 gradient_boosting (none, early_stop=off)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.03162215674470446,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 1,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
        ]
    elif data_id == "helena":
        INITIAL_CONFIGS_VALUES = [
            # 2 gradient_boosting (weighting, early_stop=off)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.09546265146045475,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 2,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 3 k_nearest_neighbors + PCA (weighting)
            {
                "classifier": "sklearn.neighbors.KNeighborsClassifier",
                "preprocessor": "sklearn.decomposition.PCA",
                "KNeighborsClassifier__n_neighbors": 4,
                "KNeighborsClassifier__p": 2,
                "KNeighborsClassifier__weights": "distance",
                # PCA params to satisfy your ConfigSpace
                "PCA__svd_solver": "randomized",
                "PCA__iterated_power": 6,
            },

            # 4 gradient_boosting (weighting, early_stop=valid)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.09750328007832798,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 25 -> 20 (cap)
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 5 gradient_boosting (weighting, early_stop=train)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.04634380160611007,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 41 -> 20 (cap)
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 6 extra_trees (none, bootstrap=True, gini)
            {
                "classifier": "sklearn.ensemble.ExtraTreesClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "ExtraTreesClassifier__bootstrap": True,
                "ExtraTreesClassifier__criterion": "gini",
                "ExtraTreesClassifier__max_features": 0.37705188916038523,
                "ExtraTreesClassifier__min_samples_leaf": 2,
                "ExtraTreesClassifier__min_samples_split": 3,
            },

            # 7 extra_trees (none, bootstrap=False, entropy)
            {
                "classifier": "sklearn.ensemble.ExtraTreesClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "ExtraTreesClassifier__bootstrap": False,
                "ExtraTreesClassifier__criterion": "entropy",
                "ExtraTreesClassifier__max_features": 0.6128603428070196,
                "ExtraTreesClassifier__min_samples_leaf": 1,
                "ExtraTreesClassifier__min_samples_split": 3,
            },

            # 8 gradient_boosting (none, early_stop=train)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.05972079854295879,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 2,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 9 gaussian_nb (weighting)
            {
                "classifier": "sklearn.naive_bayes.GaussianNB",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
            },

            # 10 random_forest + PCA (none, gini)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.decomposition.PCA",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "gini",
                "RandomForestClassifier__max_features": 0.9331254454871041,
                "RandomForestClassifier__min_samples_leaf": 2,
                "RandomForestClassifier__min_samples_split": 20,
                # PCA params to satisfy your ConfigSpace
                "PCA__svd_solver": "randomized",
                "PCA__iterated_power": 6,
            },

            # 11 gradient_boosting (none, early_stop=off)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.16262682406125173,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 12 decision_tree (weighting)
            {
                "classifier": "sklearn.tree.DecisionTreeClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "DecisionTreeClassifier__criterion": "gini",
                "DecisionTreeClassifier__max_depth": 5,  # derived from your factor
                "DecisionTreeClassifier__min_samples_split": 20,
                "DecisionTreeClassifier__min_samples_leaf": 4,
            },

            # 13 random_forest (none, bootstrap=True, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.926283631486858,
                "RandomForestClassifier__min_samples_leaf": 7,
                "RandomForestClassifier__min_samples_split": 2,
            },

            # 14 extra_trees (none, bootstrap=False, entropy)
            {
                "classifier": "sklearn.ensemble.ExtraTreesClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "ExtraTreesClassifier__bootstrap": False,
                "ExtraTreesClassifier__criterion": "entropy",
                "ExtraTreesClassifier__max_features": 0.6128603428070196,
                "ExtraTreesClassifier__min_samples_leaf": 1,
                "ExtraTreesClassifier__min_samples_split": 3,
            },

            # 15 random_forest (weighting, bootstrap=False, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.41808321658160696,
                "RandomForestClassifier__min_samples_leaf": 1,
                "RandomForestClassifier__min_samples_split": 4,
            },

            # 16 extra_trees (none, bootstrap=True, gini)
            {
                "classifier": "sklearn.ensemble.ExtraTreesClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "ExtraTreesClassifier__bootstrap": True,
                "ExtraTreesClassifier__criterion": "gini",
                "ExtraTreesClassifier__max_features": 0.37705188916038523,
                "ExtraTreesClassifier__min_samples_leaf": 2,
                "ExtraTreesClassifier__min_samples_split": 3,
            },

            # 17 k_nearest_neighbors (weighting)
            {
                "classifier": "sklearn.neighbors.KNeighborsClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "KNeighborsClassifier__n_neighbors": 2,
                "KNeighborsClassifier__p": 2,
                "KNeighborsClassifier__weights": "uniform",
            },

            # 18 gradient_boosting (none, early_stop=off)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.03162215674470446,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 1,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 19 random_forest (weighting, bootstrap=False, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.7323115919225983,
                "RandomForestClassifier__min_samples_leaf": 15,
                "RandomForestClassifier__min_samples_split": 6,
            },

            # 20 random_forest (weighting, bootstrap=True, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.6779841015398226,
                "RandomForestClassifier__min_samples_leaf": 14,
                "RandomForestClassifier__min_samples_split": 14,
            },

            # 21 random_forest (none, bootstrap=False, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.7708651296941559,
                "RandomForestClassifier__min_samples_leaf": 13,
                "RandomForestClassifier__min_samples_split": 2,
            },
        ]
    elif data_id == "jannis":
        INITIAL_CONFIGS_VALUES = [
            # 2  gradient_boosting (weighting, early_stop=valid)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.09750328007832798,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 25 -> 20 (cap)
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 3 gradient_boosting (weighting, early_stop=off, quantile→StandardScaler)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.09546265146045475,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 2,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 4 random_forest (none, bootstrap=True, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.09884140378258977,
                "RandomForestClassifier__min_samples_leaf": 6,
                "RandomForestClassifier__min_samples_split": 13,
            },

            # 5 random_forest + PCA (none, gini)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.decomposition.PCA",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "gini",
                "RandomForestClassifier__max_features": 0.9331254454871041,
                "RandomForestClassifier__min_samples_leaf": 2,
                "RandomForestClassifier__min_samples_split": 20,
                # PCA params to satisfy your ConfigSpace
                "PCA__svd_solver": "randomized",
                "PCA__iterated_power": 6,
            },

            # 6 k_nearest_neighbors + PCA (weighting)
            {
                "classifier": "sklearn.neighbors.KNeighborsClassifier",
                "preprocessor": "sklearn.decomposition.PCA",
                "KNeighborsClassifier__n_neighbors": 4,
                "KNeighborsClassifier__p": 2,
                "KNeighborsClassifier__weights": "distance",
                # PCA params to satisfy your ConfigSpace
                "PCA__svd_solver": "randomized",
                "PCA__iterated_power": 6,
            },

            # 7 gradient_boosting (weighting, early_stop=train)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.04634380160611007,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 41 -> 20 (cap)
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 8 gradient_boosting (weighting, early_stop=off)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.11042628136263043,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 22 -> 20 (cap)
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 9 gradient_boosting (none, early_stop=train)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.05972079854295879,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 2,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 10 gaussian_nb (weighting)
            {
                "classifier": "sklearn.naive_bayes.GaussianNB",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
            },

            # 11 random_forest (none, bootstrap=True, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.926283631486858,
                "RandomForestClassifier__min_samples_leaf": 7,
                "RandomForestClassifier__min_samples_split": 2,
            },

            # 12  gradient_boosting (none, early_stop=off)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.16262682406125173,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 13 extra_trees (none, bootstrap=False, entropy)
            {
                "classifier": "sklearn.ensemble.ExtraTreesClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "ExtraTreesClassifier__bootstrap": False,
                "ExtraTreesClassifier__criterion": "entropy",
                "ExtraTreesClassifier__max_features": 0.6128603428070196,
                "ExtraTreesClassifier__min_samples_leaf": 1,
                "ExtraTreesClassifier__min_samples_split": 3,
            },

            # 14 decision_tree (weighting)
            {
                "classifier": "sklearn.tree.DecisionTreeClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "DecisionTreeClassifier__criterion": "gini",
                "DecisionTreeClassifier__max_depth": 5,  # derived from factor
                "DecisionTreeClassifier__min_samples_split": 20,
                "DecisionTreeClassifier__min_samples_leaf": 4,
            },

            # 15 extra_trees (none, bootstrap=True, gini)
            {
                "classifier": "sklearn.ensemble.ExtraTreesClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "ExtraTreesClassifier__bootstrap": True,
                "ExtraTreesClassifier__criterion": "gini",
                "ExtraTreesClassifier__max_features": 0.37705188916038523,
                "ExtraTreesClassifier__min_samples_leaf": 2,
                "ExtraTreesClassifier__min_samples_split": 3,
            },

            # 16 random_forest (weighting, bootstrap=False, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.41808321658160696,
                "RandomForestClassifier__min_samples_leaf": 1,
                "RandomForestClassifier__min_samples_split": 4,
            },

            # 17 k_nearest_neighbors (weighting)
            {
                "classifier": "sklearn.neighbors.KNeighborsClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "KNeighborsClassifier__n_neighbors": 2,
                "KNeighborsClassifier__p": 2,
                "KNeighborsClassifier__weights": "uniform",
            },

            # 18 gradient_boosting (none, early_stop=off)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.03162215674470446,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 1,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 19 random_forest (weighting, bootstrap=False, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.7323115919225983,
                "RandomForestClassifier__min_samples_leaf": 15,
                "RandomForestClassifier__min_samples_split": 6,
            },

            # 20 random_forest (weighting, bootstrap=True, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.6779841015398226,
                "RandomForestClassifier__min_samples_leaf": 14,
                "RandomForestClassifier__min_samples_split": 14,
            },

            # 21 random_forest (none, bootstrap=False, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.7708651296941559,
                "RandomForestClassifier__min_samples_leaf": 13,
                "RandomForestClassifier__min_samples_split": 2,
            },
        ]
    elif data_id == "robert":
        INITIAL_CONFIGS_VALUES = [
            # 2 gradient_boosting (weighting, early_stop=train)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.04634380160611007,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 41 -> 20 (cap)
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 3 k_nearest_neighbors + PCA (weighting)
            {
                "classifier": "sklearn.neighbors.KNeighborsClassifier",
                "preprocessor": "sklearn.decomposition.PCA",
                "KNeighborsClassifier__n_neighbors": 4,
                "KNeighborsClassifier__p": 2,
                "KNeighborsClassifier__weights": "distance",
                # PCA params to satisfy your ConfigSpace
                "PCA__svd_solver": "randomized",
                "PCA__iterated_power": 6,
            },

            # 4  gradient_boosting (weighting, early_stop=off)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",  # quantile→standardized
                "GradientBoostingClassifier__learning_rate": 0.09546265146045475,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 2,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 5 gradient_boosting (weighting, early_stop=valid)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",  # minmax→standardized
                "GradientBoostingClassifier__learning_rate": 0.09750328007832798,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 25 -> 20 (cap)
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 6  random_forest (weighting, bootstrap=False, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.7323115919225983,
                "RandomForestClassifier__min_samples_leaf": 15,
                "RandomForestClassifier__min_samples_split": 6,
            },

            # 7 extra_trees (none, bootstrap=True, gini)
            {
                "classifier": "sklearn.ensemble.ExtraTreesClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",  # quantile→standardized
                "ExtraTreesClassifier__bootstrap": True,
                "ExtraTreesClassifier__criterion": "gini",
                "ExtraTreesClassifier__max_features": 0.37705188916038523,
                "ExtraTreesClassifier__min_samples_leaf": 2,
                "ExtraTreesClassifier__min_samples_split": 3,
            },

            # 8  random_forest (weighting, bootstrap=True, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.6779841015398226,
                "RandomForestClassifier__min_samples_leaf": 14,
                "RandomForestClassifier__min_samples_split": 14,
            },

            # 9 random_forest (none, bootstrap=False, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",  # quantile→standardized
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.7708651296941559,
                "RandomForestClassifier__min_samples_leaf": 13,
                "RandomForestClassifier__min_samples_split": 2,
            },

            # 10 gaussian_nb (weighting)
            {
                "classifier": "sklearn.naive_bayes.GaussianNB",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
            },

            # 11 gradient_boosting (none, early_stop=train)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",  # minmax→standardized
                "GradientBoostingClassifier__learning_rate": 0.05972079854295879,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 2,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 12 random_forest (none, bootstrap=True, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.926283631486858,
                "RandomForestClassifier__min_samples_leaf": 7,
                "RandomForestClassifier__min_samples_split": 2,
            },

            # 13 decision_tree (weighting)
            {
                "classifier": "sklearn.tree.DecisionTreeClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "DecisionTreeClassifier__criterion": "gini",
                "DecisionTreeClassifier__max_depth": 5,  # derived from factor
                "DecisionTreeClassifier__min_samples_split": 20,
                "DecisionTreeClassifier__min_samples_leaf": 4,
            },

            # 14 random_forest (none, bootstrap=True, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.09884140378258977,
                "RandomForestClassifier__min_samples_leaf": 6,
                "RandomForestClassifier__min_samples_split": 13,
            },

            # 15 gradient_boosting (none, early_stop=off)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",  # robust→standardized
                "GradientBoostingClassifier__learning_rate": 0.16262682406125173,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 16 random_forest + PCA (none, gini)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.decomposition.PCA",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "gini",
                "RandomForestClassifier__max_features": 0.9331254454871041,
                "RandomForestClassifier__min_samples_leaf": 2,
                "RandomForestClassifier__min_samples_split": 20,
                # PCA params to satisfy your ConfigSpace
                "PCA__svd_solver": "randomized",
                "PCA__iterated_power": 6,
            },

            # 17 extra_trees (none, bootstrap=False, entropy)
            {
                "classifier": "sklearn.ensemble.ExtraTreesClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "ExtraTreesClassifier__bootstrap": False,
                "ExtraTreesClassifier__criterion": "entropy",
                "ExtraTreesClassifier__max_features": 0.6128603428070196,
                "ExtraTreesClassifier__min_samples_leaf": 1,
                "ExtraTreesClassifier__min_samples_split": 3,
            },

            # 18 gradient_boosting (weighting, early_stop=off)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.11042628136263043,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 22 -> 20 (cap)
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 19 random_forest (weighting, bootstrap=False, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.41808321658160696,
                "RandomForestClassifier__min_samples_leaf": 1,
                "RandomForestClassifier__min_samples_split": 4,
            },

            # 20 k_nearest_neighbors (weighting)
            {
                "classifier": "sklearn.neighbors.KNeighborsClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "KNeighborsClassifier__n_neighbors": 2,
                "KNeighborsClassifier__p": 2,
                "KNeighborsClassifier__weights": "uniform",
            },

            # 21 gradient_boosting (none, early_stop=off)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.03162215674470446,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 1,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
        ]
    elif data_id == "jasmine":
        INITIAL_CONFIGS_VALUES = [
            # 2  gradient_boosting (weighting, early_stop=train)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.04634380160611007,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 41 -> 20 (cap)
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 3  k_nearest_neighbors + PCA (weighting)
            {
                "classifier": "sklearn.neighbors.KNeighborsClassifier",
                "preprocessor": "sklearn.decomposition.PCA",
                "KNeighborsClassifier__n_neighbors": 4,
                "KNeighborsClassifier__p": 2,
                "KNeighborsClassifier__weights": "distance",
                # PCA defaults to satisfy your ConfigSpace
                "PCA__svd_solver": "randomized",
                "PCA__iterated_power": 6,
            },

            # 4  gradient_boosting (weighting, early_stop=valid)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.09750328007832798,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 25 -> 20 (cap)
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 5  gradient_boosting (weighting, early_stop=off)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.09546265146045475,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 2,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 6  extra_trees (none, bootstrap=True, gini)
            {
                "classifier": "sklearn.ensemble.ExtraTreesClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "ExtraTreesClassifier__bootstrap": True,
                "ExtraTreesClassifier__criterion": "gini",
                "ExtraTreesClassifier__max_features": 0.37705188916038523,
                "ExtraTreesClassifier__min_samples_leaf": 2,
                "ExtraTreesClassifier__min_samples_split": 3,
            },

            # 7  extra_trees (none, bootstrap=False, entropy)
            {
                "classifier": "sklearn.ensemble.ExtraTreesClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "ExtraTreesClassifier__bootstrap": False,
                "ExtraTreesClassifier__criterion": "entropy",
                "ExtraTreesClassifier__max_features": 0.6128603428070196,
                "ExtraTreesClassifier__min_samples_leaf": 1,
                "ExtraTreesClassifier__min_samples_split": 3,
            },

            # 8 gradient_boosting (none, early_stop=train, low depth)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.05972079854295879,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 2,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 9 gaussian_nb (weighting)
            {
                "classifier": "sklearn.naive_bayes.GaussianNB",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
            },

            # 10 random_forest (none, bootstrap=True, entropy, tiny max_features)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.09884140378258977,
                "RandomForestClassifier__min_samples_leaf": 6,
                "RandomForestClassifier__min_samples_split": 13,
            },

            # 11 random_forest (none, bootstrap=True, entropy, large max_features)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.926283631486858,
                "RandomForestClassifier__min_samples_leaf": 7,
                "RandomForestClassifier__min_samples_split": 2,
            },

            # 12 gradient_boosting (none, early_stop=off, high learning_rate)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.16262682406125173,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 13 random_forest + PCA (none, gini)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.decomposition.PCA",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "gini",
                "RandomForestClassifier__max_features": 0.9331254454871041,
                "RandomForestClassifier__min_samples_leaf": 2,
                "RandomForestClassifier__min_samples_split": 20,
                # PCA defaults to satisfy your ConfigSpace
                "PCA__svd_solver": "randomized",
                "PCA__iterated_power": 6,
            },

            # 14 decision_tree (weighting)
            {
                "classifier": "sklearn.tree.DecisionTreeClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "DecisionTreeClassifier__criterion": "gini",
                "DecisionTreeClassifier__max_depth": 5,  # derived from your factor
                "DecisionTreeClassifier__min_samples_split": 20,
                "DecisionTreeClassifier__min_samples_leaf": 4,
            },

            # 15 gradient_boosting (weighting, early_stop=off)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.11042628136263043,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 22 -> 20 (cap)
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 16 random_forest (weighting, bootstrap=False, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.41808321658160696,
                "RandomForestClassifier__min_samples_leaf": 1,
                "RandomForestClassifier__min_samples_split": 4,
            },

            # 17 k_nearest_neighbors (weighting)
            {
                "classifier": "sklearn.neighbors.KNeighborsClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "KNeighborsClassifier__n_neighbors": 2,
                "KNeighborsClassifier__p": 2,
                "KNeighborsClassifier__weights": "uniform",
            },

            # 18 random_forest (weighting, bootstrap=False, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.7323115919225983,
                "RandomForestClassifier__min_samples_leaf": 15,
                "RandomForestClassifier__min_samples_split": 6,
            },

            # 19 random_forest (weighting, bootstrap=True, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.6779841015398226,
                "RandomForestClassifier__min_samples_leaf": 14,
                "RandomForestClassifier__min_samples_split": 14,
            },

            # 20 random_forest (none, bootstrap=False, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.7708651296941559,
                "RandomForestClassifier__min_samples_leaf": 13,
                "RandomForestClassifier__min_samples_split": 2,
            },

            # 21 gradient_boosting (none, early_stop=off)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.03162215674470446,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 1,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
        ]
    elif data_id == "philippine":
        INITIAL_CONFIGS_VALUES = [
            # id 1
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "gini",
                "RandomForestClassifier__max_features": 0.5,
                "RandomForestClassifier__min_samples_leaf": 1,
                "RandomForestClassifier__min_samples_split": 2,
            },
            # id 2
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.03162215674470446,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 1,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
            # id 3
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.09546265146045475,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 2,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
            # id 4
            {
                "classifier": "sklearn.neighbors.KNeighborsClassifier",
                "preprocessor": "sklearn.decomposition.PCA",
                "KNeighborsClassifier__n_neighbors": 4,
                "KNeighborsClassifier__p": 2,
                "KNeighborsClassifier__weights": "distance",
                "PCA__svd_solver": "randomized",
                "PCA__iterated_power": 6,
            },
            # id 5
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.04634380160611007,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 41 → clip to 20
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
            # id 6
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.09750328007832798,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 25 → 20
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
            # id 7
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.11042628136263043,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 22 → 20
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
            # id 8
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.09884140378258977,
                "RandomForestClassifier__min_samples_leaf": 6,
                "RandomForestClassifier__min_samples_split": 13,
            },
            # id 9
            {
                "classifier": "sklearn.naive_bayes.GaussianNB",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
            },
            # id 10
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.05972079854295879,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 2,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
            # id 11
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.decomposition.PCA",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "gini",
                "RandomForestClassifier__max_features": 0.9331254454871041,
                "RandomForestClassifier__min_samples_leaf": 2,
                "RandomForestClassifier__min_samples_split": 20,
                "PCA__svd_solver": "randomized",
                "PCA__iterated_power": 6,
            },
            # id 12
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.16262682406125173,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },
            # id 13
            {
                "classifier": "sklearn.tree.DecisionTreeClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "DecisionTreeClassifier__criterion": "gini",
                "DecisionTreeClassifier__max_depth": 5,
                "DecisionTreeClassifier__min_samples_split": 20,
                "DecisionTreeClassifier__min_samples_leaf": 4,
            },
            # id 14
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.926283631486858,
                "RandomForestClassifier__min_samples_leaf": 7,
                "RandomForestClassifier__min_samples_split": 2,
            },
            # id 15
            {
                "classifier": "sklearn.ensemble.ExtraTreesClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "ExtraTreesClassifier__bootstrap": False,
                "ExtraTreesClassifier__criterion": "entropy",
                "ExtraTreesClassifier__max_features": 0.6128603428070196,
                "ExtraTreesClassifier__min_samples_leaf": 1,
                "ExtraTreesClassifier__min_samples_split": 3,
            },
            # id 16
            {
                "classifier": "sklearn.ensemble.ExtraTreesClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "ExtraTreesClassifier__bootstrap": True,
                "ExtraTreesClassifier__criterion": "gini",
                "ExtraTreesClassifier__max_features": 0.37705188916038523,
                "ExtraTreesClassifier__min_samples_leaf": 2,
                "ExtraTreesClassifier__min_samples_split": 3,
            },
            # id 17
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.41808321658160696,
                "RandomForestClassifier__min_samples_leaf": 1,
                "RandomForestClassifier__min_samples_split": 4,
            },
            # id 18
            {
                "classifier": "sklearn.neighbors.KNeighborsClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "KNeighborsClassifier__n_neighbors": 2,
                "KNeighborsClassifier__p": 2,
                "KNeighborsClassifier__weights": "uniform",
            },
            # id 19
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.7323115919225983,
                "RandomForestClassifier__min_samples_leaf": 15,
                "RandomForestClassifier__min_samples_split": 6,
            },
            # id 20
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.6779841015398226,
                "RandomForestClassifier__min_samples_leaf": 14,
                "RandomForestClassifier__min_samples_split": 14,
            },
            # id 21
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.7708651296941559,
                "RandomForestClassifier__min_samples_leaf": 13,
                "RandomForestClassifier__min_samples_split": 2,
            },
        ]
    elif data_id == "albert":
        INITIAL_CONFIGS_VALUES = [
            # 2  gradient_boosting (weighting, early_stop=valid)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.09750328007832798,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 25 -> 20 (cap)
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 3 gradient_boosting (weighting, early_stop=off, quantile→StandardScaler)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.09546265146045475,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 2,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 4 random_forest (none, bootstrap=True, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.09884140378258977,
                "RandomForestClassifier__min_samples_leaf": 6,
                "RandomForestClassifier__min_samples_split": 13,
            },

            # 5 random_forest + PCA (none, gini)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.decomposition.PCA",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "gini",
                "RandomForestClassifier__max_features": 0.9331254454871041,
                "RandomForestClassifier__min_samples_leaf": 2,
                "RandomForestClassifier__min_samples_split": 20,
                # PCA params to satisfy your ConfigSpace
                "PCA__svd_solver": "randomized",
                "PCA__iterated_power": 6,
            },

            # 6 k_nearest_neighbors + PCA (weighting)
            {
                "classifier": "sklearn.neighbors.KNeighborsClassifier",
                "preprocessor": "sklearn.decomposition.PCA",
                "KNeighborsClassifier__n_neighbors": 4,
                "KNeighborsClassifier__p": 2,
                "KNeighborsClassifier__weights": "distance",
                # PCA params to satisfy your ConfigSpace
                "PCA__svd_solver": "randomized",
                "PCA__iterated_power": 6,
            },

            # 7 gradient_boosting (weighting, early_stop=train)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.04634380160611007,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 41 -> 20 (cap)
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 8 gradient_boosting (weighting, early_stop=off)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.11042628136263043,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,  # 22 -> 20 (cap)
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 9 gradient_boosting (none, early_stop=train)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.05972079854295879,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 2,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 10 gaussian_nb (weighting)
            {
                "classifier": "sklearn.naive_bayes.GaussianNB",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
            },

            # 11 random_forest (none, bootstrap=True, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.926283631486858,
                "RandomForestClassifier__min_samples_leaf": 7,
                "RandomForestClassifier__min_samples_split": 2,
            },

            # 12  gradient_boosting (none, early_stop=off)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.16262682406125173,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 20,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 13 extra_trees (none, bootstrap=False, entropy)
            {
                "classifier": "sklearn.ensemble.ExtraTreesClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "ExtraTreesClassifier__bootstrap": False,
                "ExtraTreesClassifier__criterion": "entropy",
                "ExtraTreesClassifier__max_features": 0.6128603428070196,
                "ExtraTreesClassifier__min_samples_leaf": 1,
                "ExtraTreesClassifier__min_samples_split": 3,
            },

            # 14 decision_tree (weighting)
            {
                "classifier": "sklearn.tree.DecisionTreeClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "DecisionTreeClassifier__criterion": "gini",
                "DecisionTreeClassifier__max_depth": 5,  # derived from factor
                "DecisionTreeClassifier__min_samples_split": 20,
                "DecisionTreeClassifier__min_samples_leaf": 4,
            },

            # 15 extra_trees (none, bootstrap=True, gini)
            {
                "classifier": "sklearn.ensemble.ExtraTreesClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "ExtraTreesClassifier__bootstrap": True,
                "ExtraTreesClassifier__criterion": "gini",
                "ExtraTreesClassifier__max_features": 0.37705188916038523,
                "ExtraTreesClassifier__min_samples_leaf": 2,
                "ExtraTreesClassifier__min_samples_split": 3,
            },

            # 16 random_forest (weighting, bootstrap=False, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.41808321658160696,
                "RandomForestClassifier__min_samples_leaf": 1,
                "RandomForestClassifier__min_samples_split": 4,
            },

            # 17 k_nearest_neighbors (weighting)
            {
                "classifier": "sklearn.neighbors.KNeighborsClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "KNeighborsClassifier__n_neighbors": 2,
                "KNeighborsClassifier__p": 2,
                "KNeighborsClassifier__weights": "uniform",
            },

            # 18 gradient_boosting (none, early_stop=off)
            {
                "classifier": "sklearn.ensemble.GradientBoostingClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "GradientBoostingClassifier__learning_rate": 0.03162215674470446,
                "GradientBoostingClassifier__max_depth": 3,
                "GradientBoostingClassifier__min_samples_leaf": 1,
                "GradientBoostingClassifier__min_samples_split": 2,
                "GradientBoostingClassifier__subsample": 1.0,
                "GradientBoostingClassifier__max_features": 0.8,
            },

            # 19 random_forest (weighting, bootstrap=False, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.7323115919225983,
                "RandomForestClassifier__min_samples_leaf": 15,
                "RandomForestClassifier__min_samples_split": 6,
            },

            # 20 random_forest (weighting, bootstrap=True, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": True,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.6779841015398226,
                "RandomForestClassifier__min_samples_leaf": 14,
                "RandomForestClassifier__min_samples_split": 14,
            },

            # 21 random_forest (none, bootstrap=False, entropy)
            {
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "preprocessor": "sklearn.preprocessing.StandardScaler",
                "RandomForestClassifier__bootstrap": False,
                "RandomForestClassifier__criterion": "entropy",
                "RandomForestClassifier__max_features": 0.7708651296941559,
                "RandomForestClassifier__min_samples_leaf": 13,
                "RandomForestClassifier__min_samples_split": 2,
            },
        ]

    for d in INITIAL_CONFIGS_VALUES:
        if d.get("preprocessor") == "sklearn.decomposition.PCA":
            d.setdefault("PCA__svd_solver", "randomized")
            d.setdefault("PCA__iterated_power", 6)

    # Build validated Configuration objects
    # Digits
    full_cs = get_tpot_configspace_classifiers_for_SMAC4AC_full()

    INITIAL_CONFIGS = [
        Configuration(full_cs, values=adapt_meta_config_to_full_cs(d, full_cs))
        for d in INITIAL_CONFIGS_VALUES
    ]

    return INITIAL_CONFIGS
    #INITIAL_CONFIGS = [Configuration(configspace, values=d) for d in INITIAL_CONFIGS_VALUES]
    #return INITIAL_CONFIGS


