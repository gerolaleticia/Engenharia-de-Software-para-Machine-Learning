# Engenharia-de-Software-para-Machine-Learning
Modelo regressivo desenvolvido orientado à objetos e seguindo boas práticas da Engenharia de Software e deployment.

## Previsão de emissão de poluente para classificação de veículos

##### O time de negócio da marca Piat deseja agrupar os veículos da empresa que tenham uma menor pegada de carbono, facilitando assim ações de marketing com este subgrupo.


*   **User Story:** "Como diretor de marketing, quero realizar a previsão de emissão de CO2 para saber o acumulado de emissão de poluente de cada veículo e decidir se ele pertence à classe de veículos sustentáveis da Piat".

###### O Co2 predictor é um software desenvolvido para gerar estes resultados para o time, a partir de um modelo de Machine Learning selecionado com AutoML. 


### Para execução do pipeline de treinamento
Uso: caso seja necessário re-treino do modelo

-executar script run_pipeline.py, que performará a ordem de execução abaixo:

    -loader.py (realiza a ingestão dos dados)   
    -pre_processor.py (pre-processamento dos dados)   
    -model_trainning.py (seleção e treinamento do melhor modelo com base na métrica r-quadrado)  
    -model_export.py (exporta o melhor model em arquivo pickle)
    -model_evaluator.py (retorna o desempenho do modelo)
    
    
### Para execução do pipeline de produção
Uso: execução mensal passando o arquivo com novos dados.

-executar script run_prod_pipeline.py, que performará a ordem de execução abaixo:

    -loader.py (ingestão dos dados)   
    -deploy_pre_processor.py (pre-processamento dos dados)   
    -deploy_load_model.py (carrega o modelo treinado) 
    -deploy_generate_output.py (gera as predições de carbono dos novos dados e exporta em .csv)
