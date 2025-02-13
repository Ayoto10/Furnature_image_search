from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import faiss
from PIL import Image
import cv2
import base64
import io

app = Flask(__name__)
CORS(app)  # السماح للـ API بالاتصال مع أي جهة

# تحميل الفهرس وقائمة الصور
index = faiss.read_index("index.faiss")
image_paths = np.load("image_paths.npy")

# استخراج الميزات من صورة الإدخال
def extract_features(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image, dtype=np.float32) / 255.0
    return image_array.flatten()

# البحث عن الصور المشابهة
def search_similar(image, top_k=5):
    query_vector = extract_features(image).reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    return [image_paths[i] for i in indices[0]]

@app.route('/search', methods=['POST'])
def search():
    try:
        file = request.files['image']
        image = Image.open(file.stream)
        results = search_similar(image)
        return jsonify({"similar_images": results})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)