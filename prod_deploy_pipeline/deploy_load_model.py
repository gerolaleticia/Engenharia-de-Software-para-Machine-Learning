import pandas as pd
import numpy as np
import pickle

class LoadModel:

    def load_trained_model(self, artifact_pkl_filename: str):
        """ Baixa o modelo salvado em arquivo pickle para aplicação nos novos dados. 
        :artifact_pkl_filename: nome do arquivo pkl
        """
        loaded_pkl_model = pickle.load(open(artifact_pkl_filename, 'rb'))
        return loaded_pkl_model