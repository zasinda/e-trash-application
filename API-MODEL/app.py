from flask import Flask, request, jsonify
import io
import os
from tensorflow import keras
import numpy as np
from PIL import Image  # Tambahkan impor ini

app = Flask(__name__)

model_trash = keras.models.load_model("model_etrash.h5")

def predict_sampah(model, request_file_key):
    if request.method == 'POST':
        if request_file_key not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files[request_file_key]
        if file.filename == "":
            return jsonify({"error": "No file"})

        img = file.read()
        img = Image.open(io.BytesIO(img)).convert("RGB")
        img = img.resize((150, 150))
        x = np.array(img) / 255.0  # Ubah img_to_array dengan np.array
        x = np.expand_dims(x, axis=0)
        
        classes = model.predict(x, batch_size=10)  # Ubah images menjadi x
        prediction = np.argmax(classes[0])

        labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]  # Tambahkan label
        result = labels[prediction]

        return result

@app.route("/")
def index():
    return "Hello World"

@app.route("/trash", methods=['POST'])
def predict():
    result = predict_sampah(model_trash, "file")
    result = {"prediction_sampah": result}
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)), debug=True)
