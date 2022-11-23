import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score


class MLEvaluator:

    def avaliar_r2_score(self, best_pipeline, X_test, Y_test):
        """ Avalia o modelo fazendo uma predição com base na métria r-quadrado
        :X_teste: features da base de teste
        :Y_teste: variável target da base de teste
        :modelo: modelo treinado
        """
        results = best_pipeline.score(X_test, Y_test)
        return print(results)