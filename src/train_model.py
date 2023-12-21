import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
import os
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.dirname(__file__))  # Navigate to 'cataract_detection_project'
train_dir = os.path.join(base_dir, 'dataset/train')
validation_dir = os.path.join(base_dir, 'dataset/validation')

# Load the pre-trained MobileNetV2 model
base_model = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
for layer in base_model.layers[:10]:
    layer.trainable = False

for layer in base_model.layers[100:]:
    layer.trainable = True

# Add custom layers for the specific task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)  # Add L2 regularization
x = Dropout(0.5)(x)  # Add dropout for regularization
output = Dense(1, activation='sigmoid')(x)  # Binary classification
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,  # Slightly reduce the rotation
    width_shift_range=0.2, ##increase to 0.2 from 0.15
    height_shift_range=0.2, ##increase to 0.2 from 0.15
    shear_range=0.2, ## add shear range
    zoom_range=0.2, ##increase to 0.2 from 0.15
    horizontal_flip=True,
    fill_mode='nearest' ##add fill mode
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load data using train_datagen.flow_from_directory(...)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(96, 96),  # Match input size of MobileNetV2
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(96, 96),
    batch_size=32,
    class_mode='binary'
)

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1) ##Reduce patience to 5
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)

# Fit the model
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Assuming 'history' is the return value from the 'fit' method of your model
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# Plot training and validation accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

model.save('modelku.h5')