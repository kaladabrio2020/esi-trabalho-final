from datetime import datetime
from ghapi.all import GhApi
import pymetrix
import math
import csv
from operator import itemgetter
import re
import git
import logging
import os
import sys
import pandas

def extract_non_patch_releases(github_token: str=None, github_owner: str="scikit-learn", 
                     github_repository: str="scikit-learn") -> list[dict]:
    """
    Essa função extrai informações sobre a lista de releases do projeto hospedado no GitHub. 
    São consideradas apenas releases que seguem a seguinte regras: 
    (1) possuem três partes: MAJOR.MINOR.PATCH; 
    (2) somente versões que não comecem em zero: MAJOR != 0; e 
    (3) Somente versões que terminam em zero: PATCH == 0.
    
    Parameters:
        github_token (str): O token de acesso do GitHub. Valor padrão é None.
        github_owner (str): O nome do owner do projeto. Valor padrão é "scikit-learn". 
        github_repository (str): O nome do repositório do projeto. Valor padrão é "scikit-learn".
    
    Returns:
        filtered_releases (list[dict]): A lista de releases em ordem decrescente de data de publicação. 
        Cada release é um dicionário com os seguintes campos: 
        “id” (identificador da relsease); 
        “tag_name” (nome da tag associada a release); e  
        “published_at”(data de publicação da relsease).
    """

    api = GhApi(token=github_token, owner=github_owner, repo=github_repository)
    releases = []
    fetched_releases = api.repos.list_releases(github_owner, github_repository, per_page=100, page=1)
    number_of_pages = math.ceil(len(fetched_releases)/100.00)
    
    for page_number in range(1, number_of_pages + 1):
        fetched_releases = api.repos.list_releases(github_owner, github_repository, per_page=100, page=page_number)
        for release in fetched_releases:
            if (not release.draft) and (not release.prerelease):
                releases.append({"id": str(release.id),
                                 "tag_name": str(release.tag_name),
                                 "published_at": release.published_at})
    filtered_releases = []
    pattern = re.compile(r'^([1-9]\d*)\.(\d+)\.(\d+)$') # Regras (1) e (2) garantidas no padrão
    for release in releases:
        match = pattern.match(release.get("tag_name"))
        if match:
            major, minor, patch = map(int, match.groups())
            if patch == 0:  # Regra (3) sendo verificada
                filtered_releases.append(release)

    return sorted(filtered_releases, key=itemgetter('published_at'), reverse=True)


def extract_post_release_timeline(releases: list[dict]=None):
    """
    Essa função determina a release alvo e o período de tempo a ser usado para extração 
    de pull requests de bug-fix. A release alvo é a antepenúltima release do projeto, 
    a data de início é a data de publicação da release alvo, a data de fim é a data de 
    publicação da última release do projeto.
    
    Parameters:
        releases (list[dict]): A lista de releases. Valor padrão é None. 
        Cada release é um dicionário com os seguintes campos: 
        “id” (identificador da relsease); 
        “tag_name” (nome da tag associada a release); e  
        “published_at”(data de publicação da relsease).
    
    Returns:
        target_release_name (str): A release alvo.
        start_date (str): A data de início da extração de dados ("%Y-%m-%d").
        end_date (str): A data de fim da extração de dados ("%Y-%m-%d").
    """
    last_release = releases[0]
    target_release = releases[2]

    target_release_name = target_release.get("tag_name")
    target_release_date = datetime.strptime(target_release.get("published_at"), "%Y-%m-%dT%H:%M:%SZ")
    last_release_date = datetime.strptime(last_release.get("published_at"), "%Y-%m-%dT%H:%M:%SZ")
    start_date = target_release_date.strftime("%Y-%m-%d")
    end_date = last_release_date.strftime("%Y-%m-%d")
    return target_release_name, start_date, end_date

def extract_bug_fix_pull_requests(github_token: str=None, github_owner: str="scikit-learn", 
                                  github_repository: str="scikit-learn", label: str="Bug", 
                                  closed_since: str=None, closed_to: str=None) -> list[str]:
    """
    Essa função faz a extração de pull requests de bug-fix no período estabelecido 
    [closed_since, closed_to]. 
    
    Parameters:
        github_token (str): O token de acesso do GitHub. Valor padrão é None.
        github_owner (str): O nome do owner do projeto. Valor padrão é "scikit-learn". 
        github_repository (str): O nome do repositório do projeto. Valor padrão é "scikit-learn".
        label (str): O rótulo a ser utilizado na filtragem. Valor padrão é "Bug".
        closed_since (str): Data que define o período inicial de pull request fechados. Valor padrão é None.
        closed_to (str): Data que define o período final de pull request fechados. Valor padrão é None.
    
    Returns:
        bug_fix_pull_request_numbers (list[str]): Lista contendo os números das pull requests selecionadas.
    """    
    date_format = "%Y-%m-%d"
    resolution_since = datetime.strptime(closed_since,date_format).strftime(date_format)
    resolution_to = datetime.strptime(closed_to,date_format).strftime(date_format)
    api = GhApi(token=github_token, owner=github_owner, repo=github_repository)
    query = f"repo:{github_owner}/{github_repository} is:pr state:closed label:{label} closed:{resolution_since}..{resolution_to}"
    bug_fix_pull_request_numbers = []
    fetched_bug_fix_pull_requests = api.search.issues_and_pull_requests(q=query, per_page=100, page=1)
    number_of_pages = math.ceil(fetched_bug_fix_pull_requests.total_count/100.00)
    
    for page_number in range(1, number_of_pages + 1):
        fetched_bug_fix_pull_requests = api.search.issues_and_pull_requests(q=query, per_page=100, page=page_number)
        pull_requests = fetched_bug_fix_pull_requests.pop("items")
        for pull_request in pull_requests:
            bug_fix_pull_request_numbers.append(str(pull_request.number))

    return list(set(bug_fix_pull_request_numbers))


def extract_bug_fix_commits(github_token: str=None, github_owner: str="scikit-learn", 
                            github_repository: str="scikit-learn", 
                            bug_fix_pull_request_numbers: list[str]=[]) -> list[str]:
    """
    Essa função faz a extração de commits de bug-fix associados às pull requests de bug-fix. 
    
    Parameters:
        github_token (str): O token de acesso do GitHub. Valor padrão é None.
        github_owner (str): O nome do owner do projeto. Valor padrão é "scikit-learn". 
        github_repository (str): O nome do repositório do projeto. Valor padrão é "scikit-learn".
        bug_fix_pull_request_numbers (list[str]): Lista contendo os números das pull requests selecionadas.  Valor padrão [].
    
    Returns:
        bug_fix_commits (list[str]): Lista contendo os hash dos commits de bug-fix.
    """
    bug_fix_commits = []
    api = GhApi(token=github_token, owner=github_owner, repo=github_repository)

    for pull_request_number in bug_fix_pull_request_numbers:
        fetched_commits = api.pulls.list_commits(pull_number=pull_request_number)
        for commit in fetched_commits:
            bug_fix_commits.append(commit.sha)
    
    return list(set(bug_fix_commits))


def extract_buggy_files(github_token: str=None, github_owner: str="scikit-learn", 
                        github_repository: str="scikit-learn", bug_fix_commits: list[str]=[], 
                        file_types=[".py"]) -> list[str]:
    """
    Essa função faz a extração de commits de bug-fix associados às pull requests de bug-fix. 
    
    Parameters:
        github_token (str): O token de acesso do GitHub. Valor padrão é None.
        github_owner (str): O nome do owner do projeto. Valor padrão é "scikit-learn". 
        github_repository (str): O nome do repositório do projeto. Valor padrão é "scikit-learn".
        bug_fix_commits (list[str]): Lista contendo os hash dos commits de bug-fix. Valor padrão [].
        file_types (list[str]): Lista de extensão dos tipos de arquivos suportados. Valor padrão [".py"].
    
    Returns:
        buggy_files (list[str]): Lista contendo os arquivos afetados pelos commits de bug-fix.
    """    
    buggy_files = []
    api = GhApi(token=github_token, owner=github_owner, repo=github_repository)

    for bug_fix in bug_fix_commits:
        fetched_commit = api.repos.get_commit(bug_fix)
        for file in fetched_commit.files:
            if str(file.filename).endswith(tuple(file_types)):
                buggy_files.append(file.filename)
    
    return buggy_files


def extract_code_metrics_and_labeling(github_owner: str="scikit-learn", github_repository: str="scikit-learn", 
                                      release_tag: str=None, buggy_files: list[str]=[]) -> str:
    """
    Essa função faz a extração das métricas de código e rotulagem nos arquivos da release alvo. 
    
    Parameters:
        github_token (str): O token de acesso do GitHub. Valor padrão é None.
        github_owner (str): O nome do owner do projeto. Valor padrão é "scikit-learn". 
        github_repository (str): O nome do repositório do projeto. Valor padrão é "scikit-learn".
        release_tag (str): Tag name da release alvo. Valor padrão None.
        buggy_files (list[str]): Lista contendo os arquivos afetados pelos commits de bug-fix.    
    
    Returns:
        code_metrics_data (list[dict]): O conjundo de métricas de código e rótulo por arquivo.  
    """
    local_repo = None
    if os.path.isdir(github_repository) and os.path.isdir(os.path.join(github_repository, '.git')):
        local_repo = git.Repo(github_repository)
    else:
        repo_url = f"https://github.com/{github_owner}/{github_repository}.git"
        local_repo = git.Repo.clone_from(repo_url, github_repository)

    local_repo.git.checkout(release_tag)
    local_repo_path = f"./{github_repository}/"
    code_metrics_data = list(pymetrix.scan_directory(local_repo_path))
    if not code_metrics_data:
        print("Nenhuma métrica foi coletada.")
        return
    
    buggy_files_full_path = []
    for i in range(len(buggy_files)):
        buggy_files_full_path.append(local_repo_path + buggy_files[i])

        for row in code_metrics_data:
            row['BUG'] = 1 if row["FILE"] in buggy_files_full_path else 0 #Aplicando o rótulo
    
    return code_metrics_data

def load_raw_dataset(release_tag: str=None, code_metrics_data:list[dict]=None) -> str:
    """
    Essa função cria um arquivo CSV contendo o conjunto de métricas extraídas por arquivo e seus respectivos rótulos.
    O nome do arquivo é iniciado pela tag alvo.
    
    Parameters:
        github_repository (str): O nome do repositório do projeto. Valor padrão é "scikit-learn".
        release_tag (str): Tag name da release alvo. Valor padrão None.
        code_metrics_data (list[dict]): O conjundo de métricas de código e rótulo por arquivo. Valor padrão None.
    
    Returns:
        dataset_file_path (str): O caminho para o dataset bruto gerado.  
    """    
    fieldnames = ['FILE', 'LOC', 'COM', 'BLK', 'NOF', 'NOC', 'APF', 'AMC', 'NER', 'NEH', 'CYC', 'MAD', 'BUG'] 
    valid_prefix = release_tag.replace(".", "_")
    dataset_file_path = f"{valid_prefix}_raw_dataset.csv"
    with open(dataset_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in code_metrics_data:
            writer.writerow(row)
    
    return dataset_file_path

def tansform_raw_dataset(dataset_file_path: str=None) -> None:
    """
    Essa função aplica algumas transformações no arquivo CSV (dataset bruto) com o intuito de remover 
    linhas que possuam as seguintes características:
    (1) trazem informações de arquivos irrelevante para a tarefa de predição 
    (e.g., arquivos de exemplos ou de documentação);
    (2) trazem valores zerados para a maioria das métricas; e
    (3) sejam valores muito fora do padrão (outliers).

    Parameters:
        dataset_file_path (str): O caminho para o dataset bruto gerado. Valor padrão None.
    
    Returns:
        transformed_file_path (str): O caminho para o dataset transformado.  
    """
    transformed_file_path = dataset_file_path.replace("raw", "trf")
    dataset = pandas.read_csv(dataset_file_path,
                              index_col=None, 
                              header=0,
                              delimiter=';')
    
    filtered_dataset = dataset[(dataset["FILE"].str.contains("/scikit-learn/examples/|/scikit-learn/doc/")==False)]
    filtered_dataset = filtered_dataset[(filtered_dataset["LOC"] > 0)]

    filtered_dataset.to_csv(transformed_file_path, 
                   sep=';', 
                   encoding='utf-8', 
                   index=False)

    return transformed_file_path

def start(token):
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='pipeline.log', encoding='utf-8', level=logging.DEBUG)
    logger.debug("[Step-1] Extraindo Releases Candidatas")
    releases = extract_non_patch_releases(github_token=token)
    logger.debug(f"\tTotal de releases {len(releases)}: {releases}")
    
    logger.debug("\n[Step-2] Extraindo Relese Alvo e Timeline")
    release, start_date, end_date = extract_post_release_timeline(releases)
    logger.debug(f"\tRelease alvo {release} no período [{start_date}, {end_date}]")

    logger.debug("\n[Step-3] Extraindo Pull Request de Bug-Fix")
    pull_request_numbers = extract_bug_fix_pull_requests(github_token=token,
                                                         closed_since=start_date,
                                                         closed_to=end_date)
    logger.debug(f"\tTotal de pull requests {len(pull_request_numbers)}: {pull_request_numbers}")
    
    logger.debug("\n[Step-4] Extraindo Commits de Bug-Fix")
    bug_fix_commits = extract_bug_fix_commits(github_token=token, 
                                              bug_fix_pull_request_numbers=pull_request_numbers)
    logger.debug(f"\tTotal de commits {len(bug_fix_commits)}: {bug_fix_commits}")
    
    logger.debug("\n[Step-5] Extraindo Arquivos Defeituosos")
    buggy_files = extract_buggy_files(github_token=token,
                                      bug_fix_commits=bug_fix_commits)
    logger.debug(f"\tTotal de arquivos {len(buggy_files)}: {buggy_files}")
    
    logger.debug("\n[Step-6] Extraindo Métricas de Código e Gerando Dataset Rotulado")
    code_metrics = extract_code_metrics_and_labeling(release_tag=release, buggy_files=buggy_files)
    logger.debug(f"\tTotal de linhas de métricas {len(code_metrics)-1}")

    logger.debug("\n[Step-7] Carregando Dataset Bruto para Arquivo")
    raw_dataset_path = load_raw_dataset(release_tag=release, code_metrics_data=code_metrics)
    logger.debug(f"\tNome do dataset criado {raw_dataset_path}")

    logger.debug("\n[Step-8] Transformando Dataset Bruto")
    transformed_dataset_path = tansform_raw_dataset(dataset_file_path=raw_dataset_path)
    logger.debug(f"\tNome do dataset transformado {transformed_dataset_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        github_token = str(sys.argv[1])
        start(github_token)
    else:
        print("Você deve prover o token de acesso à GitHub API.")    
