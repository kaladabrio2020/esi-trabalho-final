from flask import Flask, request, jsonify
import os
import numpy as np
from pathlib import Path
from service import SDPService

# Inicializa o app Flask
app = Flask(__name__)

# Configuração de diretório corrigida
BASE_DIR = Path(__file__).parent.parent.parent
MODEL_PATH = BASE_DIR / "model-RandomForest-neg-mean-squared-error.pkl"

# Inicialização com tratamento de erro
try:
    sdp_service = SDPService(str(MODEL_PATH))
    print(f"Modelo carregado com sucesso de: {MODEL_PATH}")
except Exception as e:
    print(f"Erro ao carregar modelo: {e}")
    sdp_service = None

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "mensagem": "Serviço de predição de defeitos está operacional.",
        "rota_predicao": "/predict",
        "formato_esperado": {
            "data_tuple": [-0.828975926576955, -0.19252659249677878, -0.5644251484191736, 
                          -0.6714132102320955, 1955, 2005, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -0.00059557037, 0.00062020344, 0.0038818796, 0.018665079, 0.0015334084]
        },
        "status": "OK" if sdp_service else "Modelo não carregado"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Verificar se o serviço está disponível
        if not sdp_service:
            return jsonify({"error": "Serviço indisponível - modelo não carregado."}), 500

        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON inválido ou vazio."}), 400

        data_tuple = data.get("data_tuple")

        # Validação corrigida - verificar o número correto de features
        if not data_tuple or not isinstance(data_tuple, list):
            return jsonify({"error": "data_tuple deve ser uma lista com valores numéricos."}), 400

        # O número de features parece ser 44 baseado no seu exemplo, não 31
        expected_features = len(sdp_service.X_features)
        if len(data_tuple) != expected_features:
            return jsonify({"error": f"data_tuple deve ter {expected_features} valores, recebeu {len(data_tuple)}."}), 400

        print(f"Fazendo predição com {len(data_tuple)} features...")
        result = sdp_service.predict(data_tuple)
        print(f"Resultado da predição: {result}")
        
        # Retornar o resultado corretamente
        return jsonify({"result": result.tolist() if hasattr(result, 'tolist') else [int(result[0]) if isinstance(result, (list, np.ndarray)) else int(result)]}), 200
        
    except Exception as e:
        print(f"Erro na predição: {e}")
        return jsonify({'error': f'Erro interno ao processar predição: {str(e)}'}), 500



@app.route("/showmodel", methods=["GET"])  # Changed to GET for easier testing
def show_model():
    try:
        if not sdp_service:
            return jsonify({"error": "Serviço indisponível - modelo não carregado."}), 500
            
        name, params = sdp_service.get_model()
        print(f"Modelo: {name}, Params: {params}")
        return jsonify({'name': name, 'params': params}), 200
    except Exception as e:
        print(f"Erro ao exibir modelo: {e}")
        return jsonify({'error': f'Erro ao exibir o modelo: {str(e)}'}), 500

@app.route("/features", methods=["GET"])  # Changed to GET for easier testing
def features():
    try:
        if not sdp_service:
            return jsonify({"error": "Serviço indisponível - modelo não carregado."}), 500
            
        features = sdp_service.features()
        return jsonify({'features': features}), 200
    except Exception as e:
        return jsonify({'error': f'Erro ao exibir as features: {str(e)}'}), 500

@app.route("/metricsmodel", methods=["GET"])
def show_metrics_model():
    try:
        if not sdp_service:
            return jsonify({"error": "Serviço indisponível - modelo não carregado."}), 500
            
        metrics = sdp_service.metrics()
        return jsonify(metrics), 200
    except Exception as e:
        return jsonify({'error': f'Erro ao exibir as métricas: {str(e)}'}), 500

@app.route("/newtrainmodel", methods=["POST"])
def train_model():
    try:
        if not sdp_service:
            return jsonify({"error": "Serviço indisponível - modelo não carregado."}), 500
            

        data = request.get_json()
        params = data.get("params") if data else {}
        
        result = sdp_service.train_new_model(params)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': f'Erro ao treinar o modelo: {str(e)}'}), 500
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
        