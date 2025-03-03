from flask import Flask, request, jsonify
from gradio_client import Client, handle_file
import os

app = Flask(__name__)
client = Client("Hammedalmodel/handwritten_to_text")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET"])
def home():
    return "Handwritten to Text API is running! Use POST /predict to extract text."

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)
    
    try:
        result = client.predict(
            image=handle_file(filepath),
            api_name="/predict"
        )
        os.remove(filepath)  # Cleanup
        return jsonify({"extracted_text": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
