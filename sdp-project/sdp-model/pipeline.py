import pandas
import pickle
import logging
import sys
import os
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, ShuffleSplit
from sklearn import metrics
from pathlib import Path
import numpy as np # Importar numpy para lidar com médias

# Configuração do diretório base
try:
    current_file = Path(__file__).resolve()
    # Assumindo que o script está em sdp-model e data está no mesmo nível de sdp-project
    BASE_DIR = current_file.parent.parent.parent 
except NameError:
    BASE_DIR = Path('.').resolve()

def load_dataset(dataset_path) -> pandas.DataFrame:
    """Carrega um arquivo CSV."""
    path = os.path.join(BASE_DIR, 'data', dataset_path)
    if not os.path.exists(path):
        # Fallback para o caso da estrutura de pastas ser diferente
        path = os.path.join(os.path.dirname(sys.argv[0]), dataset_path)
    print(f"Tentando carregar dataset de: {path}")
    return pandas.read_csv(path)

def save_model(model, model_name, cv_criteria):
    """Salva o modelo treinado."""
    with open(f"model-{model_name}-{cv_criteria.replace('_', '-')}.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

def load_model(file_model_path):
    """Carrega um modelo salvo."""
    return pickle.load(open(file_model_path, 'rb'))

def extract_model_metrics_scores(y_test, y_pred) -> dict:
    """Extrai métricas de regressão."""
    return {
        "mean_absolute_error": metrics.mean_absolute_error(y_test, y_pred),
        "mean_squared_error": metrics.mean_squared_error(y_test, y_pred),
        "r2_score": metrics.r2_score(y_test, y_pred)
    }

def run_experiment(dataset, x_features, y_label, models, grid_params_list, cv_criteria) -> dict:
    """Executa benchmark de modelos."""
    X = dataset[x_features]
    y = dataset[y_label]
    cv_splitter = ShuffleSplit(n_splits=5, test_size=0.25, random_state=42)
    models_info_per_fold = {}

    for i, (train_index, test_index) in enumerate(cv_splitter.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        models_info = {}
        for model_name, model_instance in models.items():
            print(f"  - Treinando {model_name} no fold {i+1}...")
            grid_model = GridSearchCV(model_instance, grid_params_list[model_name], cv=5, scoring=cv_criteria, n_jobs=-1)
            grid_model.fit(X_train, y_train)
            y_pred = grid_model.predict(X_test)
            metrics_scores = extract_model_metrics_scores(y_test, y_pred)
            models_info[model_name] = {
                "score": metrics_scores,
                "best_estimator": grid_model.best_estimator_
            }
        models_info_per_fold[i] = models_info
    return models_info_per_fold

def do_benchmark(grid_search=False, dataset_path=None, cv_criteria="neg_mean_squared_error", selected_models=["SVR", "RandomForest"]) -> dict:
    """Orquestra o benchmark."""
    dataset = load_dataset(dataset_path)
    X = dataset.drop(['date', 'price'], axis=1, errors='ignore')
    y = dataset['price']
    train_models = {
        'SVR': LinearSVR(random_state=42, max_iter=5000, dual=True), # Aumentado max_iter e setado dual
        'RandomForest': RandomForestRegressor(random_state=42),
    }
    models = {i: train_models[i] for i in train_models if i in selected_models}
    if not models:
        raise ValueError(f"Nenhum dos modelos selecionados {selected_models} está disponível em {list(train_models.keys())}")

    grid_params_list = {"SVR": {}, "RandomForest": {}}
    if grid_search:
        grid_params_list = {
            "SVR": {"C": [0.1, 1, 10]},
            "RandomForest": {"n_estimators": [50, 100], "max_depth": [None, 10]}
        }
    return run_experiment(dataset, X.columns, y.name, models, grid_params_list, cv_criteria)

# CORREÇÃO: Removido o parâmetro 'data_balance' que não é usado em regressão
def build_champion_model(dataset, x_features, y_label, model_info, cv_criteria) -> dict:
    """Constrói o modelo campeão."""
    X = dataset[x_features]
    y = dataset[y_label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Validação para garantir que o modelo não é None
    if model_info.get("instance") is None:
        raise ValueError(f"A instância do modelo para '{model_info.get('name')}' é None. Verifique os nomes dos modelos.")

    grid_model = GridSearchCV(model_info.get("instance"), model_info.get("grid_params_list"), cv=5, scoring=cv_criteria, n_jobs=-1)
    grid_model.fit(X_train, y_train)
    y_pred = grid_model.predict(X_test)
    metrics_scores = extract_model_metrics_scores(y_test, y_pred)
    save_model(grid_model.best_estimator_, model_info.get("name"), cv_criteria)
    return metrics_scores

# CORREÇÃO: Removido o parâmetro 'data_balance'
def make_model(grid_search=False, dataset_path=None, cv_criteria="neg_mean_squared_error", selected_model=None) -> dict:
    """Prepara e chama a construção do modelo campeão."""
    dataset = load_dataset(dataset_path)
    x_features = dataset.drop(['price'], axis=1, errors='ignore').columns
    y_label = 'price'
    train_models = {
        'SVR': LinearSVR(random_state=42, max_iter=5000, dual=True),
        'RandomForest': RandomForestRegressor(random_state=42),
    }
    grid_params_list = {"SVR": {}, "RandomForest": {}}
    if grid_search:
        grid_params_list = {
            "SVR": {"C": [0.1, 1, 10]},
            "RandomForest": {"n_estimators": [50, 100], "max_depth": [None, 10]}
        }
    model_info = {
        "name": selected_model,
        "instance": train_models.get(selected_model),
        "grid_params_list": grid_params_list.get(selected_model)
    }
    return build_champion_model(dataset, x_features, y_label, model_info, cv_criteria)

def select_best_model(fold_results) -> str:
    """Seleciona o melhor modelo com base no menor MSE médio."""
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
    logging.basicConfig(filename='pipeline_regression.log', filemode='w', encoding='utf-8', level=logging.DEBUG)
    
    logger.info("[Passo 1] Realizando Benchmark dos Modelos")
    print("[Passo 1] Realizando Benchmark dos Modelos...")
    
    # CORREÇÃO: Usar os nomes corretos dos modelos de regressão
    fold_results = do_benchmark(grid_search=True,
                                dataset_path=dataset_path,
                                selected_models=["SVR", "RandomForest"])

    for fold, models_info in fold_results.items():
        for model_name, info in models_info.items():
            sc = info['score']
            logger.debug(f"Fold {fold+1} - Modelo {model_name}: MSE={sc['mean_squared_error']:.4f}, MAE={sc['mean_absolute_error']:.4f}, R2={sc['r2_score']:.4f}")

    logger.info("\n[Passo 2] Selecionando o Melhor Modelo")
    print("\n[Passo 2] Selecionando o Melhor Modelo...")
    best_model_name = select_best_model(fold_results)
    logger.info(f"Melhor Modelo Selecionado: {best_model_name}")
    print(f"Melhor Modelo Selecionado: {best_model_name}")

    logger.info("\n[Passo 3] Criando o Modelo Campeão Final")
    print("\n[Passo 3] Criando o Modelo Campeão Final...")
    
    # CORREÇÃO: Removido o parâmetro 'data_balance'
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
    print(final_log_message)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_path = str(sys.argv[1])
        start(dataset_path)
    else:
        print("Erro: Você deve prover o caminho para o dataset como um argumento.")
        print("Exemplo: python seu_script.py data_transformed.csv")
