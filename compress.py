import tensorflow as tf
loaded_model = tf.keras.models.load_model('my_cnn_model_2.keras')
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('plant_disease_model.tflite', 'wb') as f:
    f.write(tflite_model)
