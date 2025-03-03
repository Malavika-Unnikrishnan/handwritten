from flask import Flask, request, jsonify
from gradio_client import Client, handle_file
import os

app = Flask(__name__)

# Initialize Gradio Client for Handwriting Recognition
client = Client("Hammedalmodel/handwritten_to_text")

# Define a folder to save uploaded images temporarily
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Handwriting Recognition API. Use /extract_text for text extraction."

@app.route("/extract_text", methods=["POST"])
def extract_text():
    # Check if an image file was uploaded
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    # Save the uploaded image
    image = request.files["image"]
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    try:
        # Use the Gradio client to process the image
        extracted_text = client.predict(
            handle_file(image_path),
            api_name="/predict"
        )
        return jsonify({"extracted_text": extracted_text}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up the uploaded file after processing
        if os.path.exists(image_path):
            os.remove(image_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
