import os
import pickle


from caps.components.HistoryGraph import extract_estimator_from_pipeline
class flaML_like:
    def __init__(self, history_id: object, directory: object = None) -> object:
        self.history_id = history_id
        if directory is None:
            directory = 'saved_graphs'
        file_path = os.path.join(directory, f"{self.history_id}.pkl")
        if os.path.exists(file_path):
            # Load the graph if it exists
            with open(file_path, 'rb') as file:
                saved_graph = pickle.load(file)
            self.flaML_metrics = saved_graph.flaML_metrics
            self.global_best_error = saved_graph.global_best_error
        else:
            self.flaML_metrics = {}
            self.global_best_error = float('inf')


    def update_flaml_metrics(self, pipeline_scores):
        """
        Updates the FlaML metrics (K0, K1, K2, D1, D2) for each learner using the given pipeline scores.

        Parameters:
        - pipeline_scores: List of dictionaries containing 'pipeline', 'score', and 'fitting_time'.
        """
        for record in pipeline_scores:
            pipeline = record['pipeline']
            if isinstance(record['score'], str):
                score = 1
            else:
                score = 1 - record['score']
            fitting_time = record['fitting_time']

            learner = extract_estimator_from_pipeline(str(pipeline))

            if learner not in self.flaML_metrics:
                self.flaML_metrics[learner] = {'K0': 0, 'K1': None, 'K2': None, 'D1': None, 'D2': None}

            # Update K0
            self.flaML_metrics[learner]['K0'] += fitting_time

            # Update D1 and D2
            if self.flaML_metrics[learner]['D1'] is None or score > self.flaML_metrics[learner]['D1']:
                self.flaML_metrics[learner]['D2'] = self.flaML_metrics[learner]['D1']
                self.flaML_metrics[learner]['D1'] = score

                # Update K1 and K2
                self.flaML_metrics[learner]['K2'] = self.flaML_metrics[learner]['K1']
                self.flaML_metrics[learner]['K1'] = self.flaML_metrics[learner]['K0']

            # Update global best error
            self.global_best_error = min(self.global_best_error, score)

        # Save metrics to file
        self.save_to_file()

    def flaml_like_selection(self, K, sklearn_pipeline_list):
        import pandas as pd

        eci_scores = []

        for idx, pipeline in enumerate(sklearn_pipeline_list):
            learner = extract_estimator_from_pipeline(str(pipeline))

            metrics = self.flaML_metrics.get(learner, {'K0': 0, 'K1': 0, 'K2': 0, 'D1': 0, 'D2': 0})
            K0, K1, K2 = metrics['K0'] or 0, metrics['K1'] or 0, metrics['K2'] or 0
            D1, D2 = metrics['D1'] or 0, metrics['D2'] or 0
            e_star = self.global_best_error  # Use the global best error

            # Compute ECI1 and ECI3
            eci1 = max(K0 - K1, K1 - K2)
            delta = D1 - D2 if D1 > D2 else 1e-6  # Avoid division by zero
            eci3 = ((D1 - e_star) * (K0 - K2)) / delta
            eci = max(eci1, eci3)

            eci_scores.append((idx, eci))

        # Sort pipelines by ECI and select the top K indices
        selected_indices_set = [idx for idx, _ in sorted(eci_scores, key=lambda x: x[1])[:K]]

        return selected_indices_set



    def save_to_file(self, directory=None):
        """
        Saves the HistoryGraph to a file named after its history_id.
        :param directory: The directory path where the file will be saved. TODO:select directory
        """
        if directory is None:
            directory = 'saved_graphs'  # Default to a 'saved_graphs' subdirectory
            if not os.path.exists(directory):
                os.makedirs(directory)  # Create the directory if it doesn't exist

        file_path = os.path.join(directory, f"{self.history_id}.pkl")
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_from_file(history_id, directory=None):
        """
        Loads a HistoryGraph from a file using its history_id.
        :param history_id: The history_id of the HistoryGraph to be loaded.
        :param directory: The directory path where the file is saved.
        :return: The loaded HistoryGraph object.
        """
        if directory is None:
            directory = 'saved_graphs'

        file_path = os.path.join(directory, f"{history_id}.pkl")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No saved file found for history_id '{history_id}' in '{directory}'")

        with open(file_path, 'rb') as file:
            return pickle.load(file)
