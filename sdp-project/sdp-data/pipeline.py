import os
import sys
import logging
import subprocess
import pandas as pd
import subprocess



import json
import time
import keras 
import tensorflow as tf
from   sklearn.mixture import GaussianMixture
from   sklearn.preprocessing import StandardScaler
import numpy as np



BASE_DIR = os.path.dirname(os.path.abspath(__file__))


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.FileHandler(BASE_DIR+f'\\logs\\data_{time.strftime("%Y_%m_%d_%H_%M_%S")}', mode='w', encoding='utf-8')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# Console 
console_handler = logging.StreamHandler(sys.stdout)
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console_handler)

def get_token(logger: logging.Logger = None):
    try:
        logger.info("Pegando o token do Kaggle\n")
        # Carregar credenciais do arquivo local
        token_path = os.path.join(BASE_DIR, '.kaggle', 'kaggle.json')
        with open(token_path, 'r') as f:
            credentials = json.load(f)

        # Definir as variáveis de ambiente
        os.environ['KAGGLE_USERNAME'] = credentials['username']
        os.environ['KAGGLE_KEY'] = credentials['key']  

        from kaggle.api.kaggle_api_extended import KaggleApi

        kaggle = KaggleApi()
        kaggle.authenticate()
        # Verificar se o token foi carregado
        logger.info("Token do Kaggle pegado com sucesso.\n")
        return kaggle
    except Exception as e:
        logger.error(e)
    
def download_dataset_from_kaggle(kaggle_dataset_id: str, download_path: str = ".", kaggle_token_pasta: str = None, logger: logging.Logger = None) -> str:
    logger.info(f"Iniciando o download do dataset: {kaggle_dataset_id}\n")
    
    # Garante que o diretório de destino exista
    os.makedirs(download_path, exist_ok=True)

    # Comando da API do Kaggle para baixar o dataset
    api = get_token(logger)
    try:
        # Executar o comando de download do Kaggle
        api.dataset_download_files(kaggle_dataset_id,path=download_path, unzip=True)

        logger.info("Download e descompactação concluídos com sucesso.\n")
        
        # O nome do arquivo CSV dentro do ZIP é geralmente 'kc_house_data.csv' para este dataset
        # Se fosse outro dataset, talvez precisássemos de uma lógica mais robusta para encontrar o .csv
        csv_filename = 'data.csv'
        extracted_csv_path = os.path.join(download_path, csv_filename)

        if not os.path.exists(extracted_csv_path):
             raise FileNotFoundError(f"Arquivo CSV esperado '{csv_filename}' não encontrado após descompactação.\n")

        return extracted_csv_path

    except FileNotFoundError:
        logger.error("Erro: O comando 'kaggle' não foi encontrado.")
        logger.error("Verifique se a biblioteca 'kaggle' está instalada e se o executável está no PATH do seu sistema.")
        return None
    
    except subprocess.CalledProcessError as e:
        logger.error("Falha ao executar o comando de download do Kaggle.")
        logger.error(f"Erro: {e.stderr}")
        return None
    
    except Exception as e:
        logger.error(f"Um erro inesperado ocorreu: {e}")
        return None


def transform_and_prepare_data(raw_csv_path: str, logger: logging.Logger = None) -> str:
    if not raw_csv_path or not os.path.exists(raw_csv_path):
        logger.error("Caminho do arquivo CSV bruto é inválido ou o arquivo não existe.\n")
        return None
        
    logger.info(f"Carregando dados de: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path)

    logger.info("Iniciando a transformação dos dados\n")

    # Função que remove colunas
    def remove_columns(df, columns_to_remove):
        logger.info(f"Removendo colunas: {columns_to_remove}")
        return df.drop(columns_to_remove, axis=1)

    def padronizacao(data):
        scaler = StandardScaler()
        logger.info("Aplicando padronização aos dados...")       
        for i in ['sqft_living', 'sqft_lot','sqft_above', 'sqft_basement']:
            data[i] = scaler.fit_transform(data[[i]].astype(float).values)
        return data
    

    def gaussian_outliers(data):
        X = data['price'].values.reshape(-1, 1)

        logger.info("Aplicando gaussian_outliers para a coluna price...")
        gm = GaussianMixture(n_components=1)
        gm.fit(X)

        densidade = gm.score_samples(X)
        logger.info(f'Percentil : 4% = das instancias serão sinalizadas como outliers')
        densidade_threshold = np.percentile(densidade, 4)

        # Máscara booleana para identificar outliers
        outlier_mask = densidade < densidade_threshold

        # Índices dos outliers
        outlier_indices = data.index[outlier_mask]

        logger.info(f'Quantidade de outliers: {len(outlier_indices)} que serão removidos')


        # Retorna DataFrame sem os outliers
        return data.drop(index=outlier_indices)


    # Função que remove casas com preços negativos
    def remove_casa_free(df):
        logger.info("Removendo instâncias de casas com preços <= 0")
        
        index = df[df['price'] <= 0].index
        
        logger.info(f"Instancias removidas: {len(index)}\n")
        return df.drop(index=index)
    
    def arrendondamento(data):
        logger.info("Aplicando arredondamento para a coluna bathrooms...")
        data['bathrooms'] = data['bathrooms'].astype(int)
        return data
        
    def dummies(data):  
        logger.info("Criando dummies para as colunas 'floors', 'waterfront', 'view' e 'condition'\n")
        return pd.get_dummies(data=data, columns=['floors', 'waterfront', 'view', 'condition'], dtype=int)
    
    def transformacao_log(data):
        logger.info("Aplicando transformação log para a coluna 'price'\n")
        data['price'] = np.log1p(data['price'])
        logger.info(" => Justificativa da transformação https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html")
        return data
    

    def tokenizer_text(data):
        logger.info("Criando tokenizer para a coluna 'location', ela é união de colunas 'street', 'city' e 'statezip'")
        
        vectorizer = keras.layers.TextVectorization(
            output_mode='int',
            output_sequence_length=10,
            standardize='lower_and_strip_punctuation',
        )
        # unindo colunas de
        data['location'] = data['street'] + ' ' + data['city'] + ' ' + data['statezip'] 

        # vetorizando
        vectorizer.adapt(data['location'].values)
        tokenizer_text = vectorizer(data['location'].values)

        return tokenizer_text, vectorizer.vocabulary_size()
    def global_mean(token, input,embedding_dim=5):
        """Aplica GlobalAveragePooling1D para reduzir a dimensionalidade do token."""

        logger.info("Aplicando GlobalAveragePooling1D para a tokens")
        embedding = keras.layers.Embedding( 
            input_dim=input,
            output_dim=embedding_dim,
        )
        token = embedding(token)

        # Aplicando GlobalAveragePooling1D
        token = keras.layers.GlobalAveragePooling1D()(token)

        return token.numpy()
    def to_colunmns(data, subset):
        logger.info("Criando colunas de location para char_token_n..")
        for i in range(subset.shape[1]):
            data[f'char_token_{i}'] = subset[:, i]

        return data
    data = remove_casa_free(df)
    data = padronizacao(data)
    data = arrendondamento(data)
    data = dummies(data)
    data = gaussian_outliers(data)
    data = transformacao_log(data)
    subset, vocabulary = tokenizer_text(data)
    subset = global_mean(subset, vocabulary)
    data  = to_colunmns(data, subset)
    data = remove_columns(data, ['date', 'street', 'city', 'statezip', 'location', 'country'])

    transformed_file_path = "./data/data_transformed.csv"
    data.to_csv(transformed_file_path, index=False)

    logger.info(f"Dataset transformado salvo em: {transformed_file_path}\n")
    return transformed_file_path

    

def start_pipeline():


    logger.info("=============================================")
    logger.info("INICIANDO O PIPELINE DE DADOS - PREÇOS DE IMÓVEIS")
    logger.info("=============================================")

    # --- MÓDULO 1: EXTRAÇÃO ---
    dataset_id = "shree1992/housedata"
    raw_data_path = download_dataset_from_kaggle(kaggle_dataset_id=dataset_id, download_path='.\\data', logger=logger)
    
    if not raw_data_path:
        logger.error("Pipeline interrompido devido a falha na extração de dados.")
        return

    # --- MÓDULO 1: TRANSFORMAÇÃO E CARGA ---
    transformed_data_path = transform_and_prepare_data(raw_csv_path=raw_data_path, logger=logger)

    if not transformed_data_path:
        logger.error("Pipeline interrompido devido a falha na transformação de dados.")
        return
        
    logger.info("=============================================")
    logger.info("PIPELINE DE DADOS CONCLUÍDO COM SUCESSO!")
    logger.info(f"O dataset final para modelagem está em: {transformed_data_path}")
    logger.info("=============================================")


if __name__ == "__main__":
    start_pipeline()