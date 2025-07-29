import pickle
import pandas as pd
import numpy as np
import subprocess
import json
from  pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
class SDPService:
    """
    Um serviço de predição de defeitos.
    """

    def __init__(self, file_model_path: str = None):
        try:
            self.model = pickle.load(open(file_model_path, 'rb'))
            
            # Definir as features esperadas
            self.X_features = [
                "sqft_living", "sqft_lot", "sqft_above", "sqft_basement", "yr_built", "yr_renovated", 
                "floors_1", "floors_2", "floors_3", "waterfront_0", "waterfront_1", "view_0", "view_1", 
                "view_2", "view_3", "view_4", "condition_1", "condition_2", "condition_3", "condition_4", 
                "condition_5", "bedrooms_0.0", "bedrooms_1.0", "bedrooms_2.0", "bedrooms_3.0", 
                "bedrooms_4.0", "bedrooms_5.0", "bedrooms_6.0", "bedrooms_7.0", "bedrooms_8.0", 
                "bedrooms_9.0", "bathrooms_0", "bathrooms_1", "bathrooms_2", "bathrooms_3", 
                "bathrooms_4", "bathrooms_5", "bathrooms_6", "bathrooms_8", "char_token_0", 
                "char_token_1", "char_token_2", "char_token_3", "char_token_4"
            ]
            self.new_model = None
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            raise e

    def predict(self, data_tuple: list = []) -> np.ndarray:
        try:
            print(f"Iniciando predição com dados: {len(data_tuple)} valores")
            
            # Verificar se o número de features está correto
            if len(data_tuple) != len(self.X_features):
                raise ValueError(f"Esperado {len(self.X_features)} features, recebido {len(data_tuple)}")
            
            # Criar DataFrame com os dados
            dataset = pd.DataFrame([data_tuple], columns=self.X_features)
            print(f"DataFrame criado com shape: {dataset.shape}")
            
            # Fazer a predição
            prediction = self.model.predict(dataset)
            print(f"Predição realizada: {prediction}")
            
            return np.expm1(prediction)
            
        except Exception as e:
            print(f"Erro na predição: {e}")
            raise e
    
    def get_model(self):
        """Método corrigido - faltava 'self'"""
        try:
            return (
                self.model.__class__.__name__,
                self.model.get_params()
            )
        except Exception as e:
            print(f"Erro ao obter informações do modelo: {e}")
            raise e
        
    def features(self):
        return self.X_features
    
    def metrics(self):
        BASE_DIR = Path(__file__).parent.parent.parent.parent
        metrics_path = BASE_DIR / "sdp-model" / "benchmark" / "metrics" / "data.json"
        with open(metrics_path, "rb") as f:
            return json.load(f)
    
    def train_new_model(self, params):
        BASE_DIR = Path(__file__).parent.parent.parent.parent.parent

        data = pd.read_csv(BASE_DIR  / "data" / "data_transformed.csv")
        X = data.drop(["price"], axis=1).values
        y = data["price"].values

        model = self.model
        model.set_params(**params)
        model.fit(X, y)
        
        pickle.dump(model, open(BASE_DIR/'sdp-project'/'sdp-service'/"model.pkl", "wb"))
        return {
            "model": model.get_params(),
            "mean_absolute_error": mean_absolute_error(y, model.predict(X)),
            "mean_squared_error": mean_squared_error(y, model.predict(X)),
            "r2_score": r2_score(y, model.predict(X))
        }
