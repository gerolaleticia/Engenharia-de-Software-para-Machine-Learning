import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class PreProcessor:

    def pre_process_data(self, data, percentual_teste, redundant_cols: list, seed=42):
        """ Realiza o pré processamento dos dados.
        :data: dataframe
        :percentual_teste: percentual dos dados definido para teste
        :seed: semente randômica
        :redundant_cols: lista de colunas que podem ser removidas do dataframe
        """

        # feature selection
        data.drop(redundant_cols, axis=1, inplace=True)

        # divisão em treino e teste
        X_train, X_test, Y_train, Y_test = self.__preparar_holdout(data,
                                                                  percentual_teste,
                                                                  seed)

        return (X_train, X_test, Y_train, Y_test)

    def __preparar_holdout(self, data, percentual_teste, seed):
        """ Divide os dados em treino e teste usando o método holdout.
        :data: dataframe
        :percentual_teste: percentual dos dados definido para teste
        :seed: semente randômica
        """
        data = data.values
        data = data.astype('float32')
        X, Y = data[:, :-1], data[:, -1]
        return train_test_split(X, Y, test_size=percentual_teste, random_state=seed)
