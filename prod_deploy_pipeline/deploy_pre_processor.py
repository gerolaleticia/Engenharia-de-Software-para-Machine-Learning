import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class PreProcessorDeploy:

    def pre_process_new_data(self, data, redundant_cols: list):
        """ Realiza o pré processamento dos dados.
        :data: dataframe com novos dados
        :redundant_cols: lista de colunas que podem ser removidas do dataframe
        """

        # feature selection
        data.drop(redundant_cols, axis=1, inplace=True)
        del data['Unnamed: 0']

        data = data.values
        data = data.astype('float32')
        return data