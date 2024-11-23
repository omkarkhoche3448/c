import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical


# Load the .npz file
data = np.load('Image classification\mnist_compressed.npz')

# Check the keys
print(data.keys())

# Extract the arrays using the correct keys
X_train = data['train_images']  # Input features for training
y_train = data['train_labels']  # Labels for training
X_test = data['test_images']    # Input features for testing
y_test = data['test_labels']    # Labels for testing

# Normalize the data (MNIST pixel values are in the range 0-255, so we scale them to 0-1)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
num_classes = len(np.unique(y_train))
print(num_classes)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
import numpy as np

# Assuming y_train and y_val are your label arrays
print("Unique classes in training data:", np.unique(y_train))
print("Number of unique classes in training data:", len(np.unique(y_train)))

print(np.unique(y_train))
print(f"Shape of y_train: {y_train.shape}, y_test: {y_test.shape}")

X_train = np.expand_dims(X_train,axis = -1)
X_test = np.expand_dims(X_test,axis = -1)
print(X_train.shape[1:])
print(X_train.shape)
print(X_test.shape)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Use 'softmax' for multi-class classification
])
model.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=1,batch_size=32)
loss,accuracy = model.evaluate(X_test,y_test)
print(loss)
print(accuracy)
import matplotlib.pyplot as plt
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Model Loss")
plt.legend()
test_image = X_test[32]
test_label = np.argmax(y_test[32])
test_image = np.expand_dims(test_image,axis=0)
prediction = np.argmax(model.predict(test_image))
plt.imshow(test_image.squeeze())
print(test_label)
print(prediction)
