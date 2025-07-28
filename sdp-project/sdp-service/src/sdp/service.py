import pickle
import pandas
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class SDPService:
    """
    Um serviço de predição de defeitos.
    """

    def __init__(self, file_model_path : str=None):
        self.model = pickle.load(open(file_model_path, 'rb'))

    def predict(self, data_tuple: list=[]) -> int:
        X_features = [
            "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "sqft_above", "sqft_basement",
            "yr_built", "yr_renovated", "floors_1.0", "floors_1.5", "floors_2.0", "floors_2.5",
            "floors_3.0", "floors_3.5", "waterfront_0", "waterfront_1", "view_0", "view_1",
            "view_2", "view_3", "view_4", "condition_1", "condition_2", "condition_3",
            "condition_4", "condition_5", "char_token_0", "char_token_1", "char_token_2",
            "char_token_3", "char_token_4"
        ]
        dataset = pandas.DataFrame([data_tuple], columns=X_features)
        X = dataset[X_features]        
        return self.model.predict(X)