import pandas as pd

class Loader:

    def load_data(self, url: str):
        """ Carrega o arquivo e retorna um DataFrame.
        :url: string com  o nome/endereço do file
        """  
        return pd.read_csv(url)