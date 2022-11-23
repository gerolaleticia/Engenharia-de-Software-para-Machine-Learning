import pandas as pd
import numpy as np

class Output:

    def create_output_dataframe(self, new_data, predicoes):
        """ Gera o dataframe de predições e concatena com os novos dados, salvando o output final em um arquivo csv.
        :predcicoes: dados de predição que o modelo gerou
        """

        output = pd.DataFrame(predicoes, columns = ['CO2_predictions'])
        output = pd.concat([new_data, output], axis=1)
        output.to_csv('predicoes.csv')
        return print(output.head())