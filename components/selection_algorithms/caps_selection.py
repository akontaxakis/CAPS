import time

import pandas as pd


def scale_to_1(series):
    x_min = series.min()
    x_max = series.max()
    return ((series - x_min) / (x_max - x_min))


def scale_to_timeout(series, timeout):
    x_min = 0
    x_max = timeout
    return (series / timeout)


def update_cost(selected_pipeline_graph, pipelines_df, timeout):
    """
    Update costs in pipelines_df after selecting a pipeline.

    Args:
        selected_pipeline_graph (nx.Graph): Graph of the selected pipeline.
        pipelines_df (pd.DataFrame): DataFrame containing remaining pipelines.
        timeout (float): Timeout value for cost scaling.

    Returns:
        pd.DataFrame: Updated DataFrame with modified costs and Scaled_Cost.
    """
    for idx, row in pipelines_df.iterrows():
        g = row['pipeline_graph']
        shared_edges = set(selected_pipeline_graph.edges()) & set(g.edges())
        if shared_edges:
            # Zero out shared edge weights
            for u, v in shared_edges:
                if 'weight' in g[u][v]:
                    g[u][v]['weight'] = 0

            # Recompute total cost as sum of edge weights
            new_cost = sum(d.get('weight', 0) for _, _, d in g.edges(data=True))
            pipelines_df.at[idx, 'Cost'] = new_cost

            # Update scaled cost only for this row
            pipelines_df.at[idx, 'Scaled_Cost'] = new_cost / timeout

    return pipelines_df


def caps_greedy_search(history, data_id, timeout, pipelines, predecessor_scores, K, l):
    """
    Select pipelines based on a weighted combination of performance and scaled cost.

    Args:
        data_id (int): Identifier for the dataset.
        timeout (float): Timeout for scaling the costs.
        pipelines (list): List of pipeline objects.
        predecessor_scores (list): List of performance scores for the pipelines.
        K (int): Number of pipelines to select.
        l (float): Lambda parameter to weight performance and cost.

    Returns:
        list: Selected pipelines.
    """
    selected_pipelines = []  # Store selected pipelines
    pipelines_selected = 0  # Track the number of selected pipelines
    pipeline_graphs = []
    # Create a DataFrame to manage pipelines, costs, and performance
    pipe_costs = []
    for pipeline in pipelines:
        request, pipeline, pipeline_graph, cost = history.estimate_and_add(data_id, pipeline)
        pipe_costs.append(cost)
        pipeline_graphs.append(pipeline_graph)

    pipelines_df = pd.DataFrame({
        'Pipeline': pipelines,
        'pipeline_graph': pipeline_graphs,
        'Cost': pipe_costs,
        'Performance': predecessor_scores
    })

    # Scale the costs
    s_time = time.time()
    pipelines_df['Scaled_Cost'] = scale_to_timeout(pipelines_df['Cost'], timeout)

    # Select pipelines based on the ratio
    while pipelines_selected < K and not pipelines_df.empty:
        # Compute the weighted ratio for each pipeline
        pipelines_df['Ratio'] = (1 - l) * pipelines_df['Performance'] - l * pipelines_df['Scaled_Cost']

        # Select the pipeline with the best ratio
        best_pipeline_idx = pipelines_df['Ratio'].idxmax()
        best_pipeline = pipelines_df.loc[best_pipeline_idx]
        selected_pipeline_graph = best_pipeline['pipeline_graph']

        # Add the selected pipeline to the result list
        selected_pipelines.append(best_pipeline['Pipeline'])
        pipelines_selected += 1

        # Remove the selected pipeline first
        pipelines_df = pipelines_df.drop(index=best_pipeline_idx).reset_index(drop=True)

        # Update costs of the remaining pipelines
        pipelines_df = update_cost(selected_pipeline_graph, pipelines_df, timeout)

    # Save the selected pipelines to file if needed
    history.save_to_file()

    selected_indices_set = [pipelines.index(pipeline) for pipeline in selected_pipelines if
                            pipeline in pipelines]

    return selected_indices_set



def caps_beam_search(history, data_id, K, timeout, pipelines, predecessor_scores, l):
    """
            Select pipelines based on a weighted combination of performance and scaled cost.

            Args:
                data_id (int): Identifier for the dataset.
                timeout (float): Timeout for scaling the costs.
                pipelines (list): List of pipeline objects.
                predecessor_scores (list): List of performance scores for the pipelines.
                K (int): Number of pipelines to select.
                l (float): Lambda parameter to weight performance and cost.

            Returns:
                list: Selected pipelines.
            """
    parser_time = cost_est_time = graph_check_time = sel_time = 0
    selected_pipelines = []  # Store selected pipelines
    pipelines_selected = 0  # Track the number of selected pipelines
    pipeline_graphs = []
    # Create a DataFrame to manage pipelines, costs, and performance
    pipe_costs = []
    for pipeline in pipelines:
        request, pipeline, pipeline_graph, cost = history.estimate_and_add(data_id, pipeline)
        pipe_costs.append(cost)
        pipeline_graphs.append(pipeline_graph)

    pipelines_df = pd.DataFrame({
        'Pipeline': pipelines,
        'pipeline_graph': pipeline_graphs,
        'Cost': pipe_costs,
        'Performance': predecessor_scores
    })

    # Scale the costs
    pipelines_df['Scaled_Cost'] = scale_to_timeout(pipelines_df['Cost'], timeout)

    beam_size = 5  # e.g., 5
    beam = []  # list of candidate sequences (each is a list of pipeline IDs)

    # initialize with top-B single pipelines
    pipelines_df['Ratio'] = (1 - l) * pipelines_df['Performance'] - l * pipelines_df['Scaled_Cost']
    top_candidates = pipelines_df.nlargest(beam_size, 'Ratio')
    for _, row in top_candidates.iterrows():
        beam.append(([row['Pipeline']], row['pipeline_graph'], row['Ratio'], {row.name}))
        # (selected pipelines, graph, score, used_indices)

    pipelines_selected = 0
    selected_pipelines = []

    while pipelines_selected < K and beam:
        new_beam = []
        for seq, graph, score, used in beam:
            # expand: try adding every unused pipeline
            for idx, row in pipelines_df.iterrows():
                if idx in used:
                    continue
                new_seq = seq + [row['Pipeline']]
                new_graph = row['pipeline_graph']
                # recompute ratio for this expansion
                new_score = score + (1 - l) * row['Performance'] - l * row['Scaled_Cost']
                new_beam.append((new_seq, new_graph, new_score, used | {idx}))
        # prune to top beam_size
        new_beam = sorted(new_beam, key=lambda x: x[2], reverse=True)[:beam_size]
        beam = new_beam
        pipelines_selected += 1

    # take the best sequence from the final beam
    best_seq, _, _, used_indices = max(beam, key=lambda x: x[2])
    selected_indices_set = list(used_indices)
    return selected_indices_set


def caps_greedy_search_overhead(history, data_id, timeout, pipelines, predecessor_scores, K, l):
    """
    Select pipelines based on a weighted combination of performance and scaled cost.

    Args:
        data_id (int): Identifier for the dataset.
        timeout (float): Timeout for scaling the costs.
        pipelines (list): List of pipeline objects.
        predecessor_scores (list): List of performance scores for the pipelines.
        K (int): Number of pipelines to select.
        l (float): Lambda parameter to weight performance and cost.

    Returns:
        list: Selected pipelines.
    """
    parser_time = cost_est_time = graph_check_time = sel_time = 0
    selected_pipelines = []  # Store selected pipelines
    pipelines_selected = 0  # Track the number of selected pipelines
    pipeline_graphs = []
    # Create a DataFrame to manage pipelines, costs, and performance
    pipe_costs = []
    for pipeline in pipelines:
        request, pipeline_graph, cost, p_time, c_time, g_time = history.estimate_and_add_time(data_id, pipeline)
        parser_time = parser_time + p_time
        cost_est_time = cost_est_time + c_time
        graph_check_time = graph_check_time + g_time
        pipe_costs.append(cost)
        pipeline_graphs.append(pipeline_graph)

    pipelines_df = pd.DataFrame({
        'Pipeline': pipelines,
        'pipeline_graph': pipeline_graphs,
        'Cost': pipe_costs,
        'Performance': predecessor_scores
    })

    # Scale the costs
    s_time = time.time()
    pipelines_df['Scaled_Cost'] = scale_to_timeout(pipelines_df['Cost'], timeout)

    # Select pipelines based on the ratio
    while pipelines_selected < K and not pipelines_df.empty:
        # Compute the weighted ratio for each pipeline
        pipelines_df['Ratio'] = (1 - l) * pipelines_df['Performance'] - l * pipelines_df['Scaled_Cost']

        # Select the pipeline with the best ratio
        best_pipeline_idx = pipelines_df['Ratio'].idxmax()
        best_pipeline = pipelines_df.loc[best_pipeline_idx]
        selected_pipeline_graph = best_pipeline['pipeline_graph']

        # Add the selected pipeline to the result list
        selected_pipelines.append(best_pipeline['Pipeline'])
        pipelines_selected += 1

        # Remove the selected pipeline first
        pipelines_df = pipelines_df.drop(index=best_pipeline_idx).reset_index(drop=True)

        # Update costs of the remaining pipelines
        pipelines_df = update_cost(selected_pipeline_graph, pipelines_df, timeout)

    # Save the selected pipelines to file if needed
    history.save_to_file()
    sel_time = time.time() - s_time
    return selected_pipelines, parser_time, cost_est_time, graph_check_time, sel_time
