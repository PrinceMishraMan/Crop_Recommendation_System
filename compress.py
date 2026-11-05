import tensorflow as tf

# Load your massive model
loaded_model = tf.keras.models.load_model('C:\\Users\\princ\\Downloads\\Programs\\Crop_Recommendation System\\my_cnn_model_2.keras')

# Initialize the converter
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)

# Apply default quantization to shrink the weights
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the tiny model
with open('plant_disease_model.tflite', 'wb') as f:
    f.write(tflite_model)