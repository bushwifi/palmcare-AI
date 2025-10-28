import numpy as np
from tensorflow import keras
from PIL import Image
import json

model = keras.models.load_model('palm_disease_model.h5')
with open('class_labels.json') as f:
    class_labels = json.load(f)

def predict(image_path):
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]
    disease = class_labels[str(class_idx)]
    
    return disease, confidence

# Usage: disease, conf = predict('test.jpg')
