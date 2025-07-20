import subprocess
import os
import pandas as pd
import zipfile
import logging

# Configuração básica do logging para acompanhar o processo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_dataset_from_kaggle(kaggle_dataset_id: str, download_path: str = ".") -> str:
    """
    Baixa e descompacta um dataset do Kaggle usando a API.

    Parameters:
        kaggle_dataset_id (str): O identificador do dataset no Kaggle (ex: 'harlfoxem/housesalesprediction').
        download_path (str): O diretório onde o arquivo será salvo.

    Returns:
        str: O caminho para o arquivo CSV descompactado.
    """
    logging.info(f"Iniciando o download do dataset: {kaggle_dataset_id}")
    
    # Garante que o diretório de destino exista
    os.makedirs(download_path, exist_ok=True)

    # Comando da API do Kaggle para baixar o dataset
    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        kaggle_dataset_id,
        "-p",
        download_path,
        "--unzip" # Descompacta o arquivo automaticamente
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info("Download e descompactação concluídos com sucesso.")
        
        # O nome do arquivo CSV dentro do ZIP é geralmente 'kc_house_data.csv' para este dataset
        # Se fosse outro dataset, talvez precisássemos de uma lógica mais robusta para encontrar o .csv
        csv_filename = 'data.csv'
        extracted_csv_path = os.path.join(download_path, csv_filename)

        if not os.path.exists(extracted_csv_path):
             raise FileNotFoundError(f"Arquivo CSV esperado '{csv_filename}' não encontrado após descompactação.")

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
        logging.error("Caminho do arquivo CSV bruto é inválido ou o arquivo não existe.")
        return None
        
    logging.info(f"Carregando dados de: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path)

    logging.info("Iniciando a transformação dos dados...")

    # Passo 1: "Tokenizar" (codificar) as colunas categóricas 'street' e 'statezip'
    # Usamos pd.factorize para uma codificação numérica simples. Ele atribui um inteiro
    # único para cada categoria em uma coluna.
    df['street_encoded'], _ = pd.factorize(df['street'])
    df['statezip_encoded'], _ = pd.factorize(df['statezip'])
    logging.info("Colunas 'street' e 'statezip' foram codificadas numericamente.")

    # Passo 2: Remover as colunas originais e outras que não serão usadas no modelo inicial.
    # A coluna 'date' é removida por simplicidade, mas poderia ser tratada
    # em uma etapa de engenharia de características mais avançada (ex: extrair ano, mês).
    columns_to_drop = ['date', 'street', 'statezip']
    df = df.drop(columns=columns_to_drop)
    logging.info(f"Colunas removidas: {columns_to_drop}")

    # Passo 3: Salvar o dataset transformado
    transformed_file_path = "data_transformed.csv"
    df.to_csv(transformed_file_path, index=False)
    logging.info(f"Dataset transformado salvo em: {transformed_file_path}")

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
    raw_data_path = download_dataset_from_kaggle(kaggle_dataset_id=dataset_id)
    
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
    start_pipeline()