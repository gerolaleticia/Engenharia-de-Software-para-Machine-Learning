import pandas as pd
import numpy as np
import pickle


class ModelExport:

    def export_best_model(self, best_pipeline):
        """ Exporta o modelo treinado em um arquivo de formato pickle
        :modelo: modelo treinado
        """
        artifact_pkl_filename = 'model.pkl'

        local_path = artifact_pkl_filename
        with open(local_path, 'wb') as model_file:
            return pickle.dump(best_pipeline, model_file)