import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
images_folder = 'object detection\Persian_Car_Plates_YOLOV8\\test\images'

def load_images_from_folder(folder):
    images = []
    image_files = sorted(os.listdir(folder))  # Sort files to maintain order

    for img_file in image_files:  # Iterate directly over filenames
        img_path = os.path.join(folder, img_file)  # Construct the full path
        img = load_img(img_path, target_size=(128, 128))  # Resize to 128x128
        images.append(img_to_array(img) / 255.0)  # Normalize images to [0, 1]

    return np.array(images)

# Load images
x = load_images_from_folder(images_folder)
print(f"Loaded {x.shape[0]} images with shape {x.shape[1:]}")

# Add less noise to images
noise_factor = 0.3  # Adjusted noise factor
x_noisy = x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
x_noisy = np.clip(x_noisy, 0., 1.)

# Split the clean and noisy images together
x_train, x_test, x_train_noisy, x_test_noisy = train_test_split(
    x, x_noisy, test_size=0.2, random_state=42
)
   # Add a channel dimension

print(x_train_noisy.shape)
print(x_test.shape)
print(x_train.shape)
print(x_test_noisy.shape)


model = keras.Sequential()
# Encoder
model.add(layers.Input(shape=(128, 128,3)))  # Change to 1 channel for grayscale
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2), padding='same'))
model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2), padding='same'))
# Decoder
model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(layers.UpSampling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.UpSampling2D((2, 2)))
model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))  # Change to 1 channel for grayscale


model.compile(optimizer='adam', loss='mean_squared_error', metrics =['accuracy'])
model.summary()
# Train the autoencoder
model.fit(x_train_noisy, x_train,
                epochs=20,
                batch_size=31,
                validation_data=(x_test_noisy, x_test))
denoised_images = model.predict(x_test_noisy)

n = 7  # Number of images to display
plt.figure(figsize=(20, 6))
for i in range(n):
    # Original images
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title("Original")
    plt.axis("off")

    # Noisy images
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_noisy[i], cmap='gray')
    plt.title("Noisy")
    plt.axis("off")

    # Denoised images
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(denoised_images[i].reshape(128, 128), cmap='gray')  # Reshape for display
    plt.title("Denoised")
    plt.axis("off")

plt.show()

