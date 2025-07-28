
import os
import time 
import sys
import pickle
import logging

import plots_ as pl
from sklearn.ensemble import (
    RandomForestRegressor
)

from sklearn.model_selection import (
    RandomizedSearchCV, 
    train_test_split, 
    ShuffleSplit
)
from sklearn.linear_model import (
    RANSACRegressor, 
    LinearRegression,
    GammaRegressor,
    PassiveAggressiveRegressor,
    ElasticNet
)

from sklearn.ensemble import (
    AdaBoostRegressor
)

import numpy as np
import pandas as pd 
from sklearn import metrics
from pathlib import Path


# Importar numpy para lidar com médias
# Configuração do diretório base
try:
    current_file = Path(__file__).resolve()
    # Assumindo que o script está em sdp-model e data está no mesmo nível de sdp-project
    BASE_DIR = current_file.parent.parent.parent 
except NameError:
    BASE_DIR = Path('.').resolve()


timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

# Caminho com data/hora
log_dir = f'benchmark/logs/logs_{timestamp}'
os.makedirs(log_dir, exist_ok=True)

# Caminho do arquivo de log
log_path_benchmark = os.path.join(log_dir, 'benchmark_regression.log')
log_path_experiment = os.path.join(log_dir, 'experiment_regression.log')
log_path_start = os.path.join(log_dir, 'start.log')

def load_dataset(dataset_path) -> pd.DataFrame:
    path = os.path.join(BASE_DIR, 'data', dataset_path)
    if not os.path.exists(path):
        # Fallback para o caso da estrutura de pastas ser diferente
        path = os.path.join(os.path.dirname(sys.argv[0]), dataset_path)
    return pd.read_csv(str(BASE_DIR)+'\\'+path)



def save_model(model, model_name, cv_criteria):
    # Salva o modelo local
    with open(f"model-{model_name}-{cv_criteria.replace('_', '-')}.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    # Salva o modelo no serviço
    with open(str(BASE_DIR)+'\\'+'sdp-project\\sdp-service'+'\\'+f"model-{model_name}-{cv_criteria.replace('_', '-')}.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

def load_model(file_model_path):
    return pickle.load(open(file_model_path, 'rb'))



def extract_model_metrics_scores(y_test, y_pred) -> dict:
    return {
        "mean_absolute_error": metrics.mean_absolute_error(y_test, y_pred),
        "mean_squared_error": metrics.mean_squared_error(y_test, y_pred),
        "r2_score": metrics.r2_score(y_test, y_pred)
    }


def todas_metricas(model, x_test, y_test):
    y_pred = model.predict(x_test)
    return {
        "mean_absolute_error": metrics.mean_absolute_error(y_test, y_pred),
        "mean_squared_error" : metrics.mean_squared_error(y_test, y_pred),
        "r2_score"           : metrics.r2_score(y_test, y_pred)
    }

def metrics_plots_best(model, X, y):
    y_pred = model.predict(X)
    pl.plot_residuals(y, y_pred)
    pl.plot_prediction_error(y, y_pred)

def metrics_plots_estimator(dicionario):
    pl.plot_estimator_metrics(dicionario)



def run_experiment(dataset, x_features, y_label, models, grid_params_list, cv_criteria) -> dict:

    # Separando as características e o alvo    
    X = dataset[x_features]
    y = dataset[y_label]

    # Definindo o splitter
    cv_splitter = ShuffleSplit(
        n_splits=5, 
        test_size=0.25, 
        random_state=42
    )

    # Definindo o dicionário
    models_info_per_fold = {}


    # Criando Logs 
    logger_experiment = logging.getLogger('experiment')
    logger_experiment.setLevel(logging.INFO)

    handler = logging.FileHandler(log_path_experiment, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    

    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    handler.setFormatter(formatter)
    logger_experiment.addHandler(handler)
    logger_experiment.addHandler(console_handler)
    #-------------------------------------------------------


    # Definindo o dicionário
    models_metrics_scores = { i: [] for i in models.keys() }

    # Iniciando o experimento
    for i, (train_index, test_index) in enumerate(cv_splitter.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        models_info = {}

        # Rodando o experimento para cada modelo
        for model_name, model_instance in models.items():
            grid_model = RandomizedSearchCV(
                estimator=model_instance, 
                param_distributions=grid_params_list[model_name], 
                cv=10, 
                scoring=cv_criteria
            )
            
            grid_model.fit(X_train, y_train)

            models_metrics_scores[model_name].append(grid_model.best_score_)

            metrics_ = todas_metricas(grid_model.best_estimator_, X_test, y_test)
            
            logger_experiment.info(f"Fold {i+1} - Modelo {model_name}: MSE={metrics_['mean_squared_error']:.4f}, MAE={metrics_['mean_absolute_error']:.4f}, R2={metrics_['r2_score']:.4f}")
            y_pred = grid_model.predict(X_test)

            metrics_scores = extract_model_metrics_scores(y_test, y_pred)
            models_info[model_name] = {
                "score": metrics_scores,
                "best_estimator": grid_model.best_estimator_
            }
        
        models_info_per_fold[i] = models_info

    metrics_plots_best(grid_model.best_estimator_, X, y)
    metrics_plots_estimator(models_metrics_scores)

    return models_info_per_fold

def do_benchmark(grid_search=False, dataset_path=None, cv_criteria="neg_mean_squared_error", selected_models=["AdaBoost", "RandomForest", "RANSAC"]) -> dict:

    logger_benchmark = logging.getLogger('benchmark')
    logger_benchmark.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path_benchmark, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    handler.setFormatter(formatter)
    logger_benchmark.addHandler(handler)
    logger_benchmark.addHandler(console_handler)

    # Carregando o dataset
    dataset = load_dataset(dataset_path)

    # Separando as características e o alvo
    X = dataset.drop(['date', 'price'], axis=1, errors='ignore')
    y = dataset['price']

    logger_benchmark.info(f"Carregando dataset de: {str(BASE_DIR)+'\\'+dataset_path}")
    logger_benchmark.info("Características:")
    logger_benchmark.info(f'  => {", ".join(X.columns)}')
    logger_benchmark.info(f"Alvo: {y.name}")

    train_models = {
        'RANSAC'       : RANSACRegressor(estimator=LinearRegression()),
        'RandomForest': RandomForestRegressor(),
        'AdaBoost'    : AdaBoostRegressor(
            estimator=PassiveAggressiveRegressor(),
        ) 
    }

    logger_benchmark.info("Modelos Disponíveis:")
    logger_benchmark.info(f'  => {", ".join(train_models.keys())}')
    
    models = {i: train_models[i] for i in train_models if i in selected_models}
    
    if not models:
        raise ValueError(f"Nenhum dos modelos selecionados {selected_models} está disponível em {list(train_models.keys())}")


    grid_params_list = {"RANSAC": {}, "RandomForest": {}, 'AdaBoost':{}}
    if grid_search:
        grid_params_list = {
            "RandomForest": {
                "n_estimators": [50, 100, 120], 
                "max_depth": [ 10, 20, 40],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4, 5, 6, 7, 8]
                },
            'AdaBoost':{
                'n_estimators':[50, 100, 150],
                'estimator':[LinearRegression(), PassiveAggressiveRegressor(), ElasticNet()],
                'loss':['linear', 'square', 'exponential'],
                'learning_rate':[0.1, 0.5, 1.0]
            },
            "RANSAC":{
                'max_trials': [100, 200, 300, 500],
                'residual_threshold': [0.01, 0.05, 0.1],
            }
        }
    logger_benchmark.info("Parâmetros de Random Search:")
    for model_name, params in grid_params_list.items():
        logger_benchmark.info(f'  => {model_name}: {", ".join(params.keys())}')

    return run_experiment(dataset, X.columns, y.name, models, grid_params_list, cv_criteria)



def build_champion_model(dataset, x_features, y_label, model_info, cv_criteria) -> dict:
    """Constrói o modelo campeão."""
    X = dataset[x_features]
    y = dataset[y_label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Validação para garantir que o modelo não é None
    if model_info.get("instance") is None:
        raise ValueError(f"A instância do modelo para '{model_info.get('name')}' é None. Verifique os nomes dos modelos.")

    grid_model = RandomizedSearchCV(model_info.get("instance"), model_info.get("grid_params_list"), cv=5, scoring=cv_criteria, n_jobs=-1)
    
    grid_model.fit(X_train, y_train)
    
    y_pred = grid_model.predict(X_test)
    
    metrics_scores = extract_model_metrics_scores(y_test, y_pred)


    save_model(grid_model.best_estimator_, model_info.get("name"), cv_criteria)
    
    return metrics_scores


def make_model(grid_search=False, dataset_path=None, cv_criteria="neg_mean_squared_error", selected_model=["RANSAC", "RandomForest", 'AdaBoost']) -> dict:

    # Carregando o dataset
    dataset    = load_dataset(dataset_path)

    # Pegando as colunas de características e o alvo
    x_features = dataset.drop(['price'], axis=1, errors='ignore').columns
    y_label = 'price'

    # Definindo os modelos de treinamento
    train_models = {
        'RANSAC': RANSACRegressor(estimator=LinearRegression()),
        'RandomForest': RandomForestRegressor(random_state=42),
        'AdaBoost': AdaBoostRegressor(
            estimator=PassiveAggressiveRegressor(),
        ) 
    }
    # Definindo os modelos selecionados
    grid_params_list = {"RANSAC": {}, "RandomForest": {}, 'AdaBoost':{}}


    if grid_search:
        grid_params_list = {
            "RandomForest": {
                "n_estimators": [50, 100, 120], 
                "max_depth": [ 10, 20, 40],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4, 5, 6, 7, 8]
            },
            'AdaBoost':{
                'n_estimators':[50, 100, 150],
                'estimator':[
                    ElasticNet(), 
                    LinearRegression(), 
                    PassiveAggressiveRegressor()
                ],
                'loss':['linear', 'square', 'exponential'],
                'learning_rate':[0.1, 0.5, 1.0]
            },
            "RANSAC":{
                'max_trials': [100, 200, 300, 500],
                'residual_threshold': [0.01, 0.05, 0.1],
            }
        }

    # Definindo o splitter
    model_info = {
        "name": selected_model,
        "instance": train_models.get(selected_model),
        "grid_params_list": grid_params_list.get(selected_model)
    }

    # Executando o experimento
    return build_champion_model(dataset, x_features, y_label, model_info, cv_criteria)




def select_best_model(fold_results) -> str:

    # Verifica se fold_results não está vazio
    if not fold_results or not fold_results[0]:
        raise ValueError("O dicionário de resultados do benchmark (fold_results) está vazio. Nenhum modelo foi treinado.")

    avg_scores = {}
    # Pega os nomes dos modelos do primeiro fold
    model_names = fold_results[0].keys()
    for model_name in model_names:
        mse_scores = [fold_results[fold][model_name]['score']['mean_squared_error'] for fold in fold_results]
        avg_scores[model_name] = np.mean(mse_scores)

    print("Média do Erro Quadrático (MSE) por modelo:", avg_scores)
    best_model_name = min(avg_scores, key=avg_scores.get)
    return best_model_name



def start(dataset_path):
    """Função principal que executa o pipeline."""
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler(log_path_start, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("[Passo 1] Realizando Benchmark dos Modelos")
    
    # CORREÇÃO: Usar os nomes corretos dos modelos de regressão
    fold_results = do_benchmark(grid_search=True,
                                dataset_path=dataset_path,
                                selected_models=["AdaBoost", "RANSAC", "RandomForest"])

    for fold, models_info in fold_results.items():
        for model_name, info in models_info.items():
            sc = info['score']
            logger.debug(f"Fold {fold+1} - Modelo {model_name}: MSE={sc['mean_squared_error']:.4f}, MAE={sc['mean_absolute_error']:.4f}, R2={sc['r2_score']:.4f}")

    logger.info("\n[Passo 2] Selecionando o Melhor Modelo")


    best_model_name = select_best_model(fold_results)

    logger.info(f"Melhor Modelo Selecionado: {best_model_name}")

    logger.info("\n[Passo 3] Criando o Modelo Campeão Final")    
    metric_scores = make_model(grid_search=True,
                               dataset_path=dataset_path,
                               selected_model=best_model_name)
    sc = metric_scores


    final_log_message = (
        f"Métricas do Modelo Campeão '{best_model_name}':\n"
        f"  - Mean Squared Error (MSE): {sc['mean_squared_error']:.4f}\n"
        f"  - Mean Absolute Error (MAE): {sc['mean_absolute_error']:.4f}\n"
        f"  - R² Score: {sc['r2_score']:.4f}"
    )
    logger.info(final_log_message)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_path = str(sys.argv[1])
        start(dataset_path)
    else:
        print("Erro: Você deve prover o caminho para o dataset como um argumento.")
        print("Exemplo: python seu_script.py data_transformed.csv")
