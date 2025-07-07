> ## **Universidade Federal do Ceará** | **Departamento de Computação**
>
> - **Curso: Bacharelado em Ciência de Dados** 
> - **Disciplina: Engenharia de Sistemas Inteligentes (CK0444 – 2025.1)** 
> - **Professor: Lincoln Souza Rocha | E-mail: lincoln@dc.ufc.br**
---
## **Instruções para Execução do Serviço de Predição de Defeitos**

1. Instale o Poetry:
```bash
pip install poetry
```
2. Instale as dependências necessárias para rodar o serviço: 
```bash
poetry install
```
3. Inicialize o serviço usando Poetry:
```bash
poetry run python -u -m sdp.app
```
4. Testando o serviço:
```bash
poetry run python -u -m unittest discover tests
```