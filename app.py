import numpy as np
import zipfile

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

# Carregue o modelo
# model = keras.models.load_model("CNN_Model")

zip_path = "CNN_Model.zip"
with zipfile.ZipFile(zip_path) as zip_file:
    model = load_model("CNN_Model")

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    # Receba os dados da requisição
    data = request.get_json()
    audio_data = data.get("audio_data")

    if audio_data:
        audio_data = np.array([audio_data])
        audio_data = audio_data.reshape(len(audio_data), -1, 1)
        # Execute a inferência do modelo nos dados recebidos
        result = model.predict(audio_data)
        result = np.argmax(result, axis=1)

        # Converta os resultados para uma resposta JSON
        response = {"result": result.tolist()}

    return jsonify(response)


if __name__ == "__main__":
    app.run()
