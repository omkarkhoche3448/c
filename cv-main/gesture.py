import numpy as np

# Load training data
X_train = np.load("gesture_detection\\train_validation\\train_gesture.npy")
y_train = np.load("gesture_detection\\train_validation\\train_gesture_labels.npy")

# Load validation data
X_val = np.load("gesture_detection\\train_validation\\validation_gesture.npy")
y_val = np.load("gesture_detection\\train_validation\\validation_gesture_labels.npy")

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")


from tensorflow.keras.utils import to_categorical

# Normalize the gesture images (if they are pixel data)
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0


# Assuming y_train and y_val are your label arrays
print("Unique classes in training data:", np.unique(y_train))
print("Number of unique classes in training data:", len(np.unique(y_train)))

print("Unique classes in validation data:", np.unique(y_val))
print("Number of unique classes in validation data:", len(np.unique(y_val)))

num_classes = len(np.unique(y_train)) 
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)

print(f"Number of classes: {num_classes}")

X_train = np.expand_dims(X_train,axis=-1)
X_val = np.expand_dims(X_val,axis = -1)

print(X_train.shape)
print(y_val.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Assuming gesture images are grayscale and have shape (height, width)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # Single neuron with sigmoid activation
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=3,
                    batch_size=32)

import matplotlib.pyplot as plt
# Example: Predicting a single gesture
test_index = 1 # Choose a sample index
test_gesture = X_val[test_index]
test_label = np.argmax(y_val[test_index])

# Expand dimensions to match input shape
test_gesture_expanded = np.expand_dims(test_gesture, axis=0)

# Predict the class
predicted_probabilities = model.predict(test_gesture_expanded)
predicted_class = np.argmax(predicted_probabilities)

print(f"True Label: {test_label}, Predicted Class: {predicted_class}")
plt.imshow(test_gesture.squeeze(),cmap='gray')



# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
plt.show()

