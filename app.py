from flask import Flask, request, jsonify
import io
from tensorflow import keras
import numpy as np
from keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)

model_trash = keras.models.load_model("model_etrash.h5")

label = []

def predict_sampah(model, request_file_key):
    if request.method == 'POST':
        if request_file_key not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files[request_file_key]
        if file.filename == "":
            return jsonify({"error": "No file"})

        img = file.read()
        img = load_img(io.BytesIO(img), target_size=(150, 150))
        x = img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])

        classes = model.predict(images, batch_size=10)
        prediction = np.argmax(classes[0])

        if prediction == 0:
            return "cardboard"
        elif prediction == 1:
            return "glass"
        elif prediction == 2:
            return "metal"
        elif prediction == 3:
            return "paper"
        elif prediction == 4:
            return "plastic"
        elif prediction == 5:
            return "trash"
        else:
            return "unknown"

@app.route("/")
def index():
    return "Hello World"

@app.route("/trash", methods=['POST'])
def predict():
    result = predict_sampah(model_trash, "file")
    prediction = result
    result = {"prediction_sampah": prediction}
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.", port=int(os.environ.get("PORT", 8080)))