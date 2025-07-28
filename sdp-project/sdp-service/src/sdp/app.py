from flask import Flask, request, jsonify
from sdp.service import SDPService

# Inicializa o app Flask
app = Flask(__name__)

MODEL_PATH = "./model-RandomForest-neg-mean-squared-error.pkl"
sdp_service = SDPService(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "mensagem": "Serviço de predição de defeitos está operacional.",
        "rota_predicao": "/predict",
        "formato_esperado": {
            "data_tuple": [100, 20, 15, 5, 2, 3, 10, 7, 5, 4, 1]
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        data_tuple = data.get("data_tuple")

        # Validação básica
        if not data_tuple or not isinstance(data_tuple, list) or len(data_tuple) != 31:
            return jsonify({"error": "data_tuple deve ser uma lista com 31 valores numéricos."}), 400


        result = sdp_service.predict(data_tuple)
        return jsonify({"result": [int(result[0])]})
    except Exception as e:
        print("Erro:", e)
        return jsonify({'error': 'Erro interno ao processar predição.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
