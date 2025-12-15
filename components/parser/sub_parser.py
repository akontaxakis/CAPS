import os
import pickle
import time
from pympler import asizeof
import inspect

from caps.components.lib import extract_platform, extract_first_two_chars, update_graph


def execute_pipeline_training(artifact_graph, dataset, pipeline, artifacts, X_train, y_train, cc):
    hs_previous = dataset + "_trainX__"
    X_temp = X_train.copy()
    y_temp = y_train.copy()
    step_full_name = hs_previous
    fitted_operator_name = ""
    for step_name, step_obj in pipeline.steps:

        platforms = []
        if "GP" in str(step_obj) or "TF" in str(step_obj) or "TR" in str(step_obj) or "GL" in str(step_obj):
            name = str(step_obj)
        else:
            name = "SK__" + str(step_obj)

        platforms.append(extract_platform(name))
        step_full_name = step_full_name + name + "__"
        hs_current = extract_first_two_chars(step_full_name)
        if hasattr(step_obj, 'fit'):
            if str(step_obj).startswith("F1ScoreCalculator") or str(step_obj).startswith("F2"):
                continue
            if str(step_obj).startswith("AccuracyCalculator"):
                continue
            if str(step_obj).startswith("ComputeAUC"):
                continue
            if str(step_obj).startswith("KS"):
                continue
            if str(step_obj).startswith("MSECalculator"):
                continue
            if str(step_obj).startswith("MAECalculator"):
                continue
            if str(step_obj).startswith("MPECalculator"):
                continue
            step_start_time = time.time()
            y_temp = y_temp[:len(X_temp)]

            args = inspect.signature(step_obj.fit).parameters
            requires_y = 'y' in args
            if requires_y:
                fitted_operator = step_obj.fit(X_temp, y_temp)
            else:
                fitted_operator = step_obj.fit(X_temp)

            step_end_time = time.time()
            step_time = step_end_time - step_start_time

            # STORE ARTIFACT
            artifacts.append(hs_current + "_fit")
            store_artifact(hs_current + "_fit", fitted_operator)

            cc = cc + step_time
            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.fit(X_temp, y_train))
            artifact_graph.add_node(hs_current + "_fit", type="fitted_operator", size=asizeof.asizeof(fitted_operator),
                                    cc=cc, frequency=1, alias = str(step_obj))
            fitted_operator_name = update_graph(artifact_graph, mem_usage, step_time, "fit", hs_previous, hs_current,
                                                platforms, step_obj)

        if hasattr(step_obj, 'transform'):
            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.transform(X_temp))
            step_start_time = time.time()
            X_temp = step_obj.transform(X_temp)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time

            # STORE ARTIFACT
            artifacts.append(hs_current + "_ftransform")
            store_artifact(hs_current + "_ftransform", X_temp)

            cc = cc + step_time
            artifact_graph.add_node(hs_current + "_ftransform", type="train", size=X_temp.size * X_temp.itemsize, cc=cc,
                                    frequency=1, alias =str(step_obj))
            artifact_graph.add_node(fitted_operator_name + "_super", type="super", size=0, cc=0, frequency=1, alias ="super")

            artifact_graph.add_edge(fitted_operator_name, fitted_operator_name + "_super", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            artifact_graph.add_edge(hs_previous, fitted_operator_name + "_super", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            hs_previous = update_graph(artifact_graph, mem_usage, step_time, "ftransform",
                                       fitted_operator_name + "_super", hs_current, platforms, step_obj)

    return artifact_graph, artifacts, pipeline, cc


def execute_pipeline_evaluation(artifact_graph, dataset, pipeline, artifacts, X_test, y_test, cc):
    X_temp = X_test.copy()
    y_temp = y_test.copy()
    hs_previous = dataset + "_testX__"
    step_full_name = hs_previous
    fitted_operator_name = ""
    for step_name, step_obj in pipeline.steps:
        platforms = []
        platforms.append(extract_platform(str(step_obj)))
        if "GP" in str(step_obj) or "TF" in str(step_obj) or "TR" in str(step_obj) or "GL" in str(step_obj) in str(
                step_obj):
            name = str(step_obj)
        else:
            name = "SK__" + str(step_obj)
        step_full_name = step_full_name + str(name) + "__"
        hs_current = extract_first_two_chars(step_full_name)
        fitted_operator_name = hs_current + "_" + "fit"

        if hasattr(step_obj, 'transform'):
            cc = artifact_graph.nodes[fitted_operator_name]['cc']
            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.transform(X_temp))
            step_start_time = time.time()
            X_temp = step_obj.transform(X_temp)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            # STORE ARTIFACT
            store_artifact(hs_current + "_tetransform", X_temp)
            artifacts.append(hs_current + "_tetransform")

            artifact_graph.add_node(hs_current + "_tetransform", type="test", size=X_temp.size * X_temp.itemsize,
                                    cc=cc + step_time, frequency=1, alias = str(step_obj))
            artifact_graph.add_node(fitted_operator_name + "_Tsuper", type="super", size=0, cc=0, frequency=1, alias = "super")

            artifact_graph.add_edge(fitted_operator_name, fitted_operator_name + "_Tsuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            artifact_graph.add_edge(hs_previous, fitted_operator_name + "_Tsuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            hs_previous = update_graph(artifact_graph, mem_usage, step_time, "tetransform",
                                       fitted_operator_name + "_Tsuper", hs_current, platforms, step_obj)

        if hasattr(step_obj, 'predict'):
            cc = artifact_graph.nodes[fitted_operator_name]['cc']
            step_start_time = time.time()
            predictions = step_obj.predict(X_temp)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            # STORE ARTIFACT
            store_artifact(hs_current + "_predict", predictions)
            artifacts.append(hs_current + "_predict")

            artifact_graph.add_node(hs_current + "_predict", type="test", size=predictions.size * predictions.itemsize,
                                    cc=cc + step_time,alias ="predictions", frequency=1)

            artifact_graph.add_node(fitted_operator_name + "_Psuper", type="super", size=0, cc=0, frequency=1, alias="super")

            artifact_graph.add_edge(fitted_operator_name, fitted_operator_name + "_Psuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            artifact_graph.add_edge(hs_previous, fitted_operator_name + "_Psuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)

            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.predict(X_temp ))
            hs_previous = update_graph(artifact_graph, mem_usage, step_time, "predict",
                                       fitted_operator_name + "_Psuper", hs_current, platforms, step_obj)
        if hasattr(step_obj, 'score'):
            if str(step_obj).startswith("F1ScoreCalculator") or str(step_obj).startswith("AccuracyCalculator") or str(
                    step_obj).startswith("MPECalculator") or str(step_obj).startswith("MSE") or str(
                step_obj).startswith("MAE") or str(step_obj).startswith("KS") or str(step_obj).startswith("VIZ") or str(
                step_obj).startswith("ComputeAUC") or str(step_obj).startswith("F2") :

                step_start_time = time.time()
                y_temp = y_temp[:len(predictions)]
                fitted_operator = step_obj.fit(y_temp)

                X_temp = fitted_operator.score(predictions)

                # STORE ARTIFACT
                store_artifact(hs_current + "_score", X_temp)
                artifacts.append(hs_current + "_score")

                step_end_time = time.time()
                step_time = step_end_time - step_start_time
                operator_string = str(step_obj)
                pos = operator_string.find('(')
                if pos != -1:
                    output_string = operator_string[:pos]
                artifact_graph.add_node(hs_current + "_score", type="score", value=X_temp, operator=output_string,
                                        size=X_temp.size * X_temp.itemsize, cc=cc, frequency=1, alias = str(output_string))

                hs_previous = update_graph(artifact_graph, mem_usage, step_time, "score", hs_previous, hs_current,
                                           platforms, step_obj)

    return artifact_graph, artifacts, hs_previous


def pipeline_training(artifact_graph, dataset, pipeline,X=1, y=None):
    cc = 0
    step_time = 0
    if X is None:
        hs_previous = dataset + "_trainX__"
    else:
        hs_previous = dataset + "_trainX__"
    step_full_name = hs_previous
    fitted_operator_name = ""
    for step_name, step_obj in pipeline.steps:
        platforms = []
        if "GP" in str(step_obj) or "TF" in str(step_obj) or "TR" in str(step_obj) or "GL" in str(step_obj):
            name = str(step_obj)
        else:
            name = "SK__" + str(step_obj)

        platforms.append(extract_platform(name))
        step_full_name = step_full_name + name + "__"
        hs_current = extract_first_two_chars(step_full_name)
        if hasattr(step_obj, 'fit'):
            if str(step_obj).startswith("F1ScoreCalculator"):
                continue
            if str(step_obj).startswith("AccuracyCalculator"):
                continue
            if str(step_obj).startswith("ComputeAUC"):
                continue
            if str(step_obj).startswith("KS"):
                continue
            if str(step_obj).startswith("MSECalculator"):
                continue
            if str(step_obj).startswith("MAECalculator"):
                continue
            if str(step_obj).startswith("MPECalculator") or str(step_obj).startswith("F2"):
                continue
            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.fit(X_temp, y_train))
            artifact_graph.add_node(hs_current + "_fit", type="fitted_operator", size=0,
                                    cc=cc, frequency=1, alias=step_name, samples=-1, features=-1)
            fitted_operator_name = update_graph(artifact_graph, mem_usage, step_time, "fit", hs_previous, hs_current,
                                                platforms, step_obj)

        if hasattr(step_obj, 'transform'):
            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.transform(X_temp))
            # tmp = X_temp.__sizeof__()
            artifact_graph.add_node(fitted_operator_name + "_super", type="super", size=0, cc=0, frequency=1,
                                    alias="super", samples=-1, features=-1)
            artifact_graph.add_node(hs_current + "_ftransform", type="train", size=0, cc=cc,
                                    frequency=1, alias=step_name, samples=-1, features=-1)
            artifact_graph.add_edge(fitted_operator_name, fitted_operator_name + "_super", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            artifact_graph.add_edge(hs_previous, fitted_operator_name + "_super", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            hs_previous = update_graph(artifact_graph, mem_usage, step_time, "ftransform",
                                       fitted_operator_name + "_super", hs_current, platforms, step_obj)

    return artifact_graph


def pipeline_evaluation(artifact_graph, dataset, pipeline):
    cc = 0
    step_time = 0
    hs_previous = dataset + "_testX__"
    step_full_name = hs_previous
    fitted_operator_name = ""
    for step_name, step_obj in pipeline.steps:
        platforms = []
        platforms.append(extract_platform(str(step_obj)))
        if "GP" in str(step_obj) or "TF" in str(step_obj) or "TR" in str(step_obj) or "GL" in str(step_obj) in str(
                step_obj):
            name = str(step_obj)
        else:
            name = "SK__" + str(step_obj)
        step_full_name = step_full_name + str(name) + "__"
        hs_current = extract_first_two_chars(step_full_name)
        fitted_operator_name = hs_current + "_" + "fit"

        if hasattr(step_obj, 'transform'):
            cc = artifact_graph.nodes[fitted_operator_name]['cc']
            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.transform(X_temp))

            artifact_graph.add_node(hs_current + "_tetransform", type="test", size=0,
                                    cc=cc + step_time, frequency=1, alias=step_name, samples=-1, features=-1)
            artifact_graph.add_node(fitted_operator_name + "_Tsuper", type="super", size=0, cc=0, frequency=1,
                                    alias="super", samples=-1, features=-1)

            artifact_graph.add_edge(fitted_operator_name, fitted_operator_name + "_Tsuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            artifact_graph.add_edge(hs_previous, fitted_operator_name + "_Tsuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            hs_previous = update_graph(artifact_graph, mem_usage, step_time, "tetransform",
                                       fitted_operator_name + "_Tsuper", hs_current, platforms, step_obj)

        if hasattr(step_obj, 'predict'):
            artifact_graph.add_node(hs_current + "_predict", type="test", size=0,
                                    cc=cc + step_time, frequency=1, alias=step_name, samples=-1, features=-1)

            artifact_graph.add_node(fitted_operator_name + "_Psuper", type="super", size=0, cc=0, frequency=1,
                                    alias="super")

            artifact_graph.add_edge(fitted_operator_name, fitted_operator_name + "_Psuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            artifact_graph.add_edge(hs_previous, fitted_operator_name + "_Psuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)

            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.predict(X_temp ))
            hs_previous = update_graph(artifact_graph, mem_usage, step_time, "predict",
                                       fitted_operator_name + "_Psuper", hs_current, platforms, step_obj)
        if hasattr(step_obj, 'score'):
            if str(step_obj).startswith("F1ScoreCalculator") or str(step_obj).startswith("AccuracyCalculator") or str(
                    step_obj).startswith("MPECalculator") or str(step_obj).startswith("MSE") or str(
                step_obj).startswith("MAE") or str(step_obj).startswith("KS") or str(step_obj).startswith("VIZ") or str(step_obj).startswith("F2"):
                artifact_graph.add_node(hs_current + "_score", type="score",
                                        size=0, cc=cc, frequency=1, alias=step_name, samples=-1, features=-1)

                hs_previous = update_graph(artifact_graph, mem_usage, step_time, "score", hs_previous, hs_current,
                                           platforms, step_obj)

    return artifact_graph, hs_previous


def store_artifact(hs_current, artifact, directory=None):
    if directory is None:
        directory = 'artifact_storage'  # Default to a 'artifact_storage' subdirectory
        if not os.path.exists(directory):
            os.makedirs(directory)
    file_path = os.path.join(directory, f"{hs_current}.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(artifact, f)
