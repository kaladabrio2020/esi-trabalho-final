import subprocess
import os
import pandas as pd
import zipfile
import logging

import keras 
import tensorflow as tf
import json

import numpy as np
# Configuração básica do logging para acompanhar o processo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_token():
    try:
        logging.info("Pegando o token do Kaggle\n")
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
        logging.info("Token do Kaggle pegado com sucesso.\n")
        return kaggle
    except Exception as e:
        logging.error(e)
    
def download_dataset_from_kaggle(kaggle_dataset_id: str, download_path: str = ".", kaggle_token_pasta: str = None) -> str:
    """
    Baixa e descompacta um dataset do Kaggle usando a API.

    Parameters:
        kaggle_dataset_id (str): O identificador do dataset no Kaggle (ex: 'harlfoxem/housesalesprediction').
        download_path (str): O diretório onde o arquivo será salvo.

    Returns:
        str: O caminho para o arquivo CSV descompactado.
    """


    logging.info(f"Iniciando o download do dataset: {kaggle_dataset_id}\n")
    
    # Garante que o diretório de destino exista
    os.makedirs(download_path, exist_ok=True)

    # Comando da API do Kaggle para baixar o dataset
    api = get_token()
    try:
        # Executar o comando de download do Kaggle
        api.dataset_download_files(kaggle_dataset_id, path=download_path, unzip=True)

        logging.info("Download e descompactação concluídos com sucesso.\n")
        
        # O nome do arquivo CSV dentro do ZIP é geralmente 'kc_house_data.csv' para este dataset
        # Se fosse outro dataset, talvez precisássemos de uma lógica mais robusta para encontrar o .csv
        csv_filename = 'data.csv'
        extracted_csv_path = os.path.join(download_path, csv_filename)

        if not os.path.exists(extracted_csv_path):
             raise FileNotFoundError(f"Arquivo CSV esperado '{csv_filename}' não encontrado após descompactação.\n")

        return extracted_csv_path

    except FileNotFoundError:
        logging.error("Erro: O comando 'kaggle' não foi encontrado.")
        logging.error("Verifique se a biblioteca 'kaggle' está instalada e se o executável está no PATH do seu sistema.")
        return None
    except subprocess.CalledProcessError as e:
        logging.error("Falha ao executar o comando de download do Kaggle.")
        logging.error(f"Erro: {e.stderr}")
        return None
    except Exception as e:
        logging.error(f"Um erro inesperado ocorreu: {e}")
        return None


def transform_and_prepare_data(raw_csv_path: str) -> str:
    """
    Carrega o dataset bruto, aplica transformações e o prepara para modelagem.
    - Codifica as colunas 'street' e 'statezip'.
    - Remove colunas não utilizadas.
    - Salva o dataset transformado.

    Parameters:
        raw_csv_path (str): O caminho para o arquivo CSV bruto.

    Returns:
        str: O caminho para o arquivo CSV transformado e salvo.
    """
    if not raw_csv_path or not os.path.exists(raw_csv_path):
        logging.error("Caminho do arquivo CSV bruto é inválido ou o arquivo não existe.\n")
        return None
        
    logging.info(f"Carregando dados de: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path)

    logging.info("Iniciando a transformação dos dados\n")

    # Função que remove colunas
    def remove_columns(df, columns_to_remove):
        logging.info(f"Removendo colunas: {columns_to_remove}")
        return df.drop(columns_to_remove, axis=1)
    from sklearn.preprocessing import StandardScaler

    def padronizacao(data):
        scaler = StandardScaler()
        logging.info("Aplicando padronização aos dados...")       
        for i in ['sqft_living', 'sqft_lot','sqft_above', 'sqft_basement']:
            data[i] = scaler.fit_transform(data[[i]].astype(float).values)
        return data
    
    from sklearn.mixture import GaussianMixture
    def gaussian_outliers(data):
        X = data['price'].values.reshape(-1, 1)

        logging.info("Aplicando gaussian_outliers para a coluna price...")
        gm = GaussianMixture(n_components=1)
        gm.fit(X)

        densidade = gm.score_samples(X)
        logging.info(f'Percentil : 4% = das instancias serão sinalizadas como outliers')
        densidade_threshold = np.percentile(densidade, 4)

        # Máscara booleana para identificar outliers
        outlier_mask = densidade < densidade_threshold

        # Índices dos outliers
        outlier_indices = data.index[outlier_mask]

        logging.info(f'Quantidade de outliers: {len(outlier_indices)} que serão removidos')


        # Retorna DataFrame sem os outliers
        return data.drop(index=outlier_indices)


    # Função que remove casas com preços negativos
    def remove_casa_free(df):
        logging.info("Removendo instâncias de casas com preços <= 0")
        
        index = df[df['price'] <= 0].index
        
        logging.info(f"Instancias removidas: {len(index)}\n")
        return df.drop(index=index)
    
    def arrendondamento(data):
        logging.info("Aplicando arredondamento para a coluna bathrooms...")
        data['bathrooms'] = data['bathrooms'].astype(int)
        return data
        
    def dummies(data):
        
        logging.info("Criando dummies para as colunas 'floors', 'waterfront', 'view' e 'condition'\n")
        return pd.get_dummies(data=data, columns=['floors', 'waterfront', 'view', 'condition'], dtype=int)
    
    import numpy as np
    def transformacao_log(data):
        logging.info("Aplicando transformação log para a coluna 'price'\n")
        data['price'] = np.log1p(data['price'])
        logging.info(" => Justificativa da transformação https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html")
        return data
    

    def tokenizer_text(data):
        logging.info("Criando tokenizer para a coluna 'location', ela é união de colunas 'street', 'city' e 'statezip'")
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
        ebedding_dim = 5
        embedding = keras.layers.Embedding( 
            input_dim=input,
            output_dim=embedding_dim,
        )
        token = embedding(token)

        # Aplicando GlobalAveragePooling1D
        token = keras.layers.GlobalAveragePooling1D()(token)

        return token.numpy()
    def to_colunmns(data, subset):
        logging.info("Criando colunas de location para char_token_n..")
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
    logging.info(f"Dataset transformado salvo em: {transformed_file_path}\n")

    return transformed_file_path

    

def start_pipeline():
    """
    Função principal que orquestra a execução do pipeline de dados.
    """
    logging.info("=============================================")
    logging.info("INICIANDO O PIPELINE DE DADOS - PREÇOS DE IMÓVEIS")
    logging.info("=============================================")

    # --- MÓDULO 1: EXTRAÇÃO ---
    dataset_id = "shree1992/housedata"
    raw_data_path = download_dataset_from_kaggle(kaggle_dataset_id=dataset_id, download_path='.\\data')
    
    if not raw_data_path:
        logging.error("Pipeline interrompido devido a falha na extração de dados.")
        return

    # --- MÓDULO 1: TRANSFORMAÇÃO E CARGA ---
    transformed_data_path = transform_and_prepare_data(raw_csv_path=raw_data_path)

    if not transformed_data_path:
        logging.error("Pipeline interrompido devido a falha na transformação de dados.")
        return
        
    logging.info("=============================================")
    logging.info("PIPELINE DE DADOS CONCLUÍDO COM SUCESSO!")
    logging.info(f"O dataset final para modelagem está em: {transformed_data_path}")
    logging.info("=============================================")


if __name__ == "__main__":
    print(BASE_DIR)
    start_pipeline()