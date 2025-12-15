import logging
from caps.components.selection_algorithms.caps_selection import caps_greedy_search, caps_beam_search
from caps.components.selection_algorithms.flaML_like import flaML_like

def CAPS_middleware(sel_algo, lamda, selection, data_id, random_state, N,timeout,sklearn_pipeline_list,predecessor_scores, logging=None):
    K = selection
    l = lamda
    G = N
    exp_id = data_id + "_" + str(random_state) + '_' + str(K) + '_' + sel_algo + '_' + str(l) + '_' + str(G)

    log_file_name = data_id + "_" + str(random_state) + '_' + str(K) + '_' + sel_algo + '_' + str(l) + '_' + str(
            G) + '.log'
    import logging
    logging.basicConfig(filename=log_file_name, level=logging.INFO)
    logging.info(f" Generation Started")
    if sel_algo == "ratio":
        from caps.components.HistoryGraph import HistoryGraph
        history = HistoryGraph(str(exp_id))
        history.add_dataset_split(data_id, 0.3)
        perf_cost = []
        for idx, sklearn_pipeline in enumerate(sklearn_pipeline_list):
            request, pipeline, cost = history.estimate_and_add(data_id, sklearn_pipeline)
            if cost > 0:
                ratio = predecessor_scores[idx] / cost
            else:
                ratio = 100000.0
            perf_cost.append(ratio)
        selected_indices_set = sorted(range(len(perf_cost)), key=lambda i: perf_cost[i], reverse=True)[:K]
    if sel_algo =="flaML_like":
        from caps.components.selection_algorithms.flaML_like import flaML_like
        flaML = flaML_like(str(exp_id))
        selected_indices_set = flaML.flaml_like_selection(K, sklearn_pipeline_list)

    if sel_algo =="caps_beam_search":
        from caps.components.HistoryGraph import HistoryGraph
        history = HistoryGraph(str(exp_id))
        history.add_dataset_split(data_id, 0.3)
        selected_indices_set = caps_beam_search(history, data_id, timeout, sklearn_pipeline_list, predecessor_scores, K, l)


    if sel_algo =="caps_greedy":
        from caps.components.HistoryGraph import HistoryGraph
        history = HistoryGraph(str(exp_id))
        history.add_dataset_split(data_id, 0.3)
        selected_indices_set = caps_greedy_search(history, data_id, timeout, sklearn_pipeline_list, predecessor_scores,
                                                  K, l)


    print("Indexes of the selected pipelines:", selected_indices_set)
    logging.info("Indexes of the selected pipelines: %s", selected_indices_set)


    sklearn_pipeline_list = [val for idx, val in enumerate(sklearn_pipeline_list) if
                                 idx in selected_indices_set]

    return selected_indices_set, sklearn_pipeline_list

def CAPS_update(sel_algo,lamda, selection,data_id,random_state,N, pipeline_scores):
    K = selection
    l = lamda
    G = N
    exp_id = data_id + "_" + str(random_state) + '_' + str(K) + '_' + sel_algo + '_' + str(l) + '_' + str(G)
    if sel_algo =="flaML_like":
        from caps.components.selection_algorithms.flaML_like import flaML_like
        flaML = flaML_like(str(exp_id))
        flaML.update_flaml_metrics(pipeline_scores)