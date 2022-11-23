import warnings
from sklearn.ensemble import ExtraTreesRegressor
from loader import Loader
from deploy_pre_processor import PreProcessorDeploy
from deploy_load_model import LoadModel
from deploy_generate_output import Output

# Configurações
warnings.filterwarnings("ignore")

# Instanciação das Classes
loader = Loader()
preprocess_deploy = PreProcessorDeploy()
generate_output = Output()
load_model = LoadModel()

# Parâmetros
redundant_cols = ['MODELYEAR','MAKE','MODEL','VEHICLECLASS','TRANSMISSION','FUELTYPE']
new_file = "brand_new_data.csv"
pkl_model_file = 'model.pkl'

def main():
    # Execução do pipeline de produção

    # carga
    new_data = loader.load_data(new_file) 

    # processamento dos novos dados
    X = preprocess_deploy.pre_process_new_data(new_data, redundant_cols)

    # load do modelo treinado
    loaded_pkl_model = load_model.load_trained_model(pkl_model_file)

    # aplicação do modelo já treinado
    predicoes = loaded_pkl_model.predict(X)

    # geração e export do arquivo de predições em csv
    generate_output.create_output_dataframe(new_data, predicoes)

if __name__ == '__main__':
    main()