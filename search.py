import os
import numpy as np
import faiss
import cv2
from PIL import Image

# مسار مجلد الصور
DATASET_FOLDER = "data_test"

# تحميل الصور وتحويلها إلى ميزات (Feature Vectors)
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image, dtype=np.float32) / 255.0
    return image_array.flatten()

# تخزين الميزات
features = []
image_paths = []

# قراءة كل الصور داخل المجلدات الفرعية
for category in os.listdir(DATASET_FOLDER):
    category_path = os.path.join(DATASET_FOLDER, category)
    if os.path.isdir(category_path):
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            try:
                feature_vector = extract_features(image_path)
                features.append(feature_vector)
                image_paths.append(image_path)
            except Exception as e:
                print(f"خطأ في الصورة {image_path}: {e}")

# تحويل الميزات إلى مصفوفة NumPy
features = np.array(features, dtype=np.float32)

# إنشاء الفهرس باستخدام FAISS
dimension = features.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(features)

# حفظ الفهرس وقائمة الصور
faiss.write_index(index, "index.faiss")
np.save("image_paths.npy", image_paths)

print("✅ تم إنشاء الفهرس بنجاح!")
