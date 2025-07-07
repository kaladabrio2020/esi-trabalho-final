from flask import Flask, request, request, jsonify
from sdp.service import SDPService

app = Flask(__name__)
sdp_service = SDPService("./model-RFC-ROC_AUC-SMOTE.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        data_tuple = data.get('data_tuple')
        result = sdp_service.predict(data_tuple)
        return jsonify({"result": [int(element) for element in result]})
    except (TypeError, ValueError) as e:
        print(e)
        return jsonify({'error': 'Invalid parameters'}), 400

if __name__ == '__main__':
    app.run(debug=True)