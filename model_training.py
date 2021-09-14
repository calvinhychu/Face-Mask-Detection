import tensorflow
import numpy as np 
# from random import randint
# from sklearn.utils import shuffle 
# from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import binary_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from keras.applications.vgg16 import VGG16
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt

train_path = 'train'
test_path = 'test'

# Data augementation to generate more images to deal with overfitting. 20% of train images will be for validation. 
image_generator = ImageDataGenerator(rotation_range=10, width_shift_range=0.2,
height_shift_range=0.2, shear_range=0.15, zoom_range=0.1,
channel_shift_range=10, horizontal_flip=True, validation_split=0.2)


train_batches = image_generator.flow_from_directory(batch_size=10,
                                                 directory=train_path,
                                                 shuffle=True,
                                                 target_size=(224, 224), 
                                                 subset="training",
                                                 classes=['Mask', 'No Mask'])

validation_batches = image_generator.flow_from_directory(batch_size=10,
                                                 directory=train_path,
                                                 shuffle=True,
                                                 target_size=(224, 224), 
                                                 subset="validation",
                                                 classes=['Mask', 'No Mask'])

test_batches = ImageDataGenerator(preprocessing_function=tensorflow.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=['Mask', 'No Mask'], batch_size=10, shuffle=False)

imgs, labels = next(train_batches)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Using VGG16 as base model
vgg = VGG16(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

# Freeze trainable params of base model
vgg.trainable = False

# Add more layers to base model
model = Sequential([
    vgg,
    Conv2D(filters=100, kernel_size=(3, 3), activation ='relu', padding = 'same'), # Extract spatial features
    MaxPool2D(pool_size=(2, 2), strides = 2), # cut our image dimension by half
    Conv2D(filters=128, kernel_size=(3, 3), activation ='relu', padding = 'same'), # Extract spatial features with more filters
    MaxPool2D(pool_size=(2, 2), strides = 2), 
    Flatten(), # Flatten tensor to 1 dimension to prepare for input for Dense layer
    Dropout(0.5), # To lessen overfitting
    Dense(units = 50, activation='relu'), 
    Dense(units=2, activation='softmax'), # 2 outputs of Mask or No Mask
])

# Stop training when val loss has stopped improving.
early_stopping_monitor = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=2, # Number of epochs with no improvement after which training will be stopped.
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True # Save model with best trained weight
)


# # Compile model with Adam as optimizer and binary_crossentropy as loss function
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=["accuracy"]) # Binary outcome

# # Train model
m = model.fit(x=train_batches, validation_data=validation_batches, batch_size=10, epochs=8, callbacks=[early_stopping_monitor], verbose=2)
model.save('saved_model/my_model11')

# Plotting training result 
plt.plot(m.history["loss"], label="train_loss")
plt.plot(m.history["accuracy"], label="train_acc")
plt.plot(m.history["val_loss"], label="val_loss")
plt.plot(m.history["val_accuracy"], label="val_acc")
plt.title("Training Summary")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


predictions = model.predict(x=test_batches, verbose=0)
cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
cm_plot_labels=['Mask', 'No Mask']
plot_confusion_matrix(cm, classes=cm_plot_labels, title='Matrix')