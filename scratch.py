import tensorflow as tf

# Load the HDF5 model
model = tf.keras.models.load_model('models/cataract_detection_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)