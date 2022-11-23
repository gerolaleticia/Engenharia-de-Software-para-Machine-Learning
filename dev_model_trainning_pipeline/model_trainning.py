import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor

class MLModel:

    def select_best_model(self, cv, X_train, Y_train):
        """ Utiliza AutoML para identificar o melhor modelo de regressão.
        :cv: define a validação cruzada
        :X_train: features da base de treino
        :Y_treino: variável target da base de treino
        """

        # define busca do melhor modelo de regressão
        model = TPOTRegressor(generations=5, population_size=50, scoring='r2', cv=cv, verbosity=2, random_state=1, n_jobs=-1)
        model.fit(X_train, Y_train)
        model.export('best_model.py')
        
        # display resultados do AutoML
        resultado = pd.DataFrame(model.evaluated_individuals_)
        resultado.columns = list(map(lambda x: x[0], resultado.columns.str.split('(')))
        return print(resultado.T)

    def model_trainning(self, X_train, Y_train):
        """ Cria pipeline de treinamento do melhor modelo encontrado na etapa de search.
        :X_train: features da base de treino
        :Y_treino: variável target da base de treino
        """
        best_pipeline = ExtraTreesRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=1, min_samples_split=5, n_estimators=100)
        best_pipeline = best_pipeline.fit(X_train, Y_train)
        return best_pipeline 