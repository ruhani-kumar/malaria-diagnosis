import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = tf.keras.models.load_model("models/malaria_finetuned_model.keras")

image = Image.open("demo/demo_2_Parasitized.jpg").resize((224, 224))
image_array = np.array(image)
image_array = preprocess_input(image_array)
image_array = np.expand_dims(image_array, axis=0)

prediction = model.predict(image_array)[0][0]
label = "Parasitized" if prediction < 0.5 else "Uninfected"
confidence = prediction if prediction < 0.5 else 1 - prediction

print(f"Label: {label}, Confidence: {confidence:.4f}")
