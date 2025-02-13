import numpy as np
import faiss
from PIL import Image
import cv2

# تحميل الفهرس وقائمة الصور
index = faiss.read_index("index.faiss")
image_paths = np.load("image_paths.npy")

# استخراج الميزات من صورة الإدخال
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image, dtype=np.float32) / 255.0
    return image_array.flatten()

# البحث عن الصور المشابهة
def search_similar(image_path, top_k=5):
    query_vector = extract_features(image_path).reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    return [image_paths[i] for i in indices[0]]

# اختبار البحث
test_image = "test_image.jpeg"  # صورة الاختبار
similar_images = search_similar(test_image)

print("🔍 الصور المشابهة:")
for img in similar_images:
    print(img)
