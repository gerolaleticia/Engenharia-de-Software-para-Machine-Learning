import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor
import warnings
from loader import Loader
from pre_processor import PreProcessor
from model_trainning import MLModel
from model_evaluator import MLEvaluator
from model_export import ModelExport

# Configurações
warnings.filterwarnings("ignore")

# Instanciação das Classes
loader = Loader()
pre_processor = PreProcessor()
model = MLModel()
performance_evaluator = MLEvaluator()
export_model = ModelExport()

# Parâmetros
url_dados = ('FuelConsumptionCo2.csv')
redundant_cols = ['MODELYEAR','MAKE','MODEL','VEHICLECLASS','TRANSMISSION','FUELTYPE']
percentual_teste = 0.2
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1) 

def main():
    # Execução do pipeline de treinamento

    # carga
    data = loader.load_data(url_dados) 

    X_train, X_test, Y_train, Y_test = pre_processor.pre_process_data(data, percentual_teste,redundant_cols) 

    # Busca e seleção do melhor modelo com Auto ML
    best_model_report = model.select_best_model(cv, X_train, Y_train)

    # Treinamento do melhor modelo
    best_pipeline = model.model_trainning(X_train, Y_train)

    # Resultados de performance considerando r2 score
    performance_evaluator.avaliar_r2_score(best_pipeline, X_test, Y_test)

    # Export do modelo treinado
    loaded_pkl_model = export_model.export_best_model(best_pipeline)
    
if __name__ == '__main__':
    main()