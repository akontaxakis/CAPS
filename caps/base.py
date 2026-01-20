import logging
from caps.components.selection_algorithms.caps_selection import caps_greedy_search, caps_beam_search
from caps.components.selection_algorithms.flaML_like import flaML_like

def CAPS_middleware(
    sel_algo, lamda, selection, data_id, random_state, N, timeout,
    sklearn_pipeline_list, predecessor_scores, logging=None
):
    K = selection
    l = lamda
    G = N

    exp_id = f"{data_id}_{random_state}_{K}_{sel_algo}_{l}_{G}"
    log_file_name = f"{exp_id}.log"

    import logging
    logging.basicConfig(filename=log_file_name, level=logging.INFO)
    logging.info("Generation Started")

    # --------------------------------------
    # valid pipelines (score > 0)
    # --------------------------------------
    valid_indices = [i for i, p in enumerate(sklearn_pipeline_list) if p != 0]

    if len(valid_indices) == 0:
        logging.warning("No valid pipelines found (all scores are 0).")
        return [], []

    # ======================================
    # RATIO
    # ======================================
    if sel_algo == "ratio":
        from caps.components.HistoryGraph import HistoryGraph

        history = HistoryGraph(str(exp_id))
        history.add_dataset_split(data_id, 0.3)

        scored = []
        for i in valid_indices:
            request, pipeline, cost = history.estimate_and_add(
                data_id, sklearn_pipeline_list[i]
            )

            if cost > 0:
                value = predecessor_scores[i] / cost
            else:
                value = -float("inf")  # penalize invalid cost

            scored.append((i, value))  # keep ORIGINAL index

        selected_indices_set = [
            i for i, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:K]
        ]

    # ======================================
    # FLAML-LIKE
    # ======================================
    elif sel_algo == "flaML_like":
        from caps.components.selection_algorithms.flaML_like import flaML_like

        flaML = flaML_like(str(exp_id))
        selected_indices_set = flaML.flaml_like_selection(
            K, [sklearn_pipeline_list[i] for i in valid_indices]
        )

        # map back to original indices
        selected_indices_set = [valid_indices[i] for i in selected_indices_set]

    # ======================================
    # CAPS BEAM SEARCH
    # ======================================
    elif sel_algo == "caps_beam_search":
        from caps.components.HistoryGraph import HistoryGraph

        history = HistoryGraph(str(exp_id))
        history.add_dataset_split(data_id, 0.3)

        selected_indices_set = caps_beam_search(
            history,
            data_id,
            timeout,
            [sklearn_pipeline_list[i] for i in valid_indices],
            [predecessor_scores[i] for i in valid_indices],
            K,
            l,
        )

        selected_indices_set = [valid_indices[i] for i in selected_indices_set]

    # ======================================
    # CAPS GREEDY
    # ======================================
    elif sel_algo == "caps_greedy":
        from caps.components.HistoryGraph import HistoryGraph

        history = HistoryGraph(str(exp_id))
        history.add_dataset_split(data_id, 0.3)

        selected_indices_set = caps_greedy_search(
            history,
            data_id,
            timeout,
            [sklearn_pipeline_list[i] for i in valid_indices],
            [predecessor_scores[i] for i in valid_indices],
            K,
            l,
        )

        selected_indices_set = [valid_indices[i] for i in selected_indices_set]

    else:
        raise ValueError(f"Unknown selection algorithm: {sel_algo}")

    logging.info("Indexes of the selected pipelines: %s", selected_indices_set)
    print("Indexes of the selected pipelines:", selected_indices_set)

    # slice ONLY at the end
    selected_pipelines = [sklearn_pipeline_list[i] for i in selected_indices_set]

    return selected_indices_set, selected_pipelines


def CAPS_update(sel_algo,lamda, selection,data_id,random_state,N, pipeline_scores):
    K = selection
    l = lamda
    G = N
    exp_id = data_id + "_" + str(random_state) + '_' + str(K) + '_' + sel_algo + '_' + str(l) + '_' + str(G)
    if sel_algo =="flaML_like":
        from caps.components.selection_algorithms.flaML_like import flaML_like
        flaML = flaML_like(str(exp_id))
        flaML.update_flaml_metrics(pipeline_scores)