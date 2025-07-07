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
        X_features = ["LOC", "COM", "BLK", "NOF", "NOC", "APF", "AMC", "NER", "NEH", "CYC", "MAD"]
        dataset = pandas.DataFrame([data_tuple], columns=X_features)
        X = dataset[X_features]        
        return self.model.predict(X)