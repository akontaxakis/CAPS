def AutoMLselection_lamda(self, data_id, K, sklearn_pipeline_list, predecessor_scores, l, Utility, timeout=None):
    import pandas as pd
    if Utility == "DIV":
        if isinstance(sklearn_pipeline_list, pd.DataFrame):
            if 'cost' in sklearn_pipeline_list.columns:
                selected_pipelines = self.select_pipelines_based_on_diversity_lamda_for_df_scaled(data_id,
                                                                                                  sklearn_pipeline_list,
                                                                                                  K,
                                                                                                  l)
                selected_indices_set = [
                    sklearn_pipeline_list[sklearn_pipeline_list['pipeline'] == pipeline].index[0]
                    for pipeline in selected_pipelines
                    if pipeline in sklearn_pipeline_list['pipeline'].values
                ]
        else:
            selected_pipelines = self.select_pipelines_based_on_diversity_lamda(data_id, sklearn_pipeline_list, K, l)
            selected_indices_set = [sklearn_pipeline_list.index(pipeline) for pipeline in selected_pipelines if
                                    pipeline in sklearn_pipeline_list]
    else:
        # self, data_id, timeout, pipelines,predecessor_scores, K, l
        selected_pipelines = self.select_pipelines_based_on_performance_lambda(data_id, timeout, sklearn_pipeline_list,
                                                                               predecessor_scores, K, l)
        selected_indices_set = [sklearn_pipeline_list.index(pipeline) for pipeline in selected_pipelines if
                                pipeline in sklearn_pipeline_list]
    return selected_indices_set
