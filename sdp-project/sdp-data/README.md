> ## **Universidade Federal do Ceará** | **Departamento de Computação**
>
> - **Curso: Bacharelado em Ciência de Dados** 
> - **Disciplina: Engenharia de Sistemas Inteligentes (CK0444 – 2025.1)** 
> - **Professor: Lincoln Souza Rocha | E-mail: lincoln@dc.ufc.br**
---
## **Instruções para Execução da Pipeline de Dados**
1. Crie um token de acesso no GitHub. Para isso, vá até "Settings" >> "Developer settings" >> "Personal access tokens" >> "Fine-grained personal access tokens" >> "Generate new token".
2. Instale o Poetry:
```bash
pip install poetry
```
3. Instale as dependências necessárias para rodar a pipeline de modelos: 
```bash
poetry install
```
4. Execute a pipeline e dados:
```bash
poetry run python pipeline.py <GitHub Access Token>
```