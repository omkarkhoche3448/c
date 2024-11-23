import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, UpSampling2D,Input,concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf

images = 'semantic segmentation\semantic segmentation\Images'
labels = 'semantic segmentation\semantic segmentation\Labels'
def load_images_and_masks(image_dir, label_dir, image_size):
    images = []
    masks = []
    image_files = sorted(os.listdir(image_dir))  # Sort to ensure matching order
    label_files = sorted(os.listdir(label_dir))  # Sort to ensure matching order

    for img_file, mask_file in zip(image_files, label_files):
        # Load and preprocess image
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(label_dir, mask_file)

        img = load_img(img_path, target_size=image_size)
        mask = load_img(mask_path, target_size=image_size, color_mode="grayscale")

        images.append(img_to_array(img) / 255.0)  # Normalize images to [0, 1]
        masks.append(img_to_array(mask) / 255.0)  # Normalize masks to [0, 1]

    return np.array(images), np.array(masks)

images, masks = load_images_and_masks(images, labels, (128,128))
num_classes = 2
masks = np.round(masks).astype("int")  # Ensure masks are binary
masks = to_categorical(masks, num_classes=num_classes)
print(masks.shape)
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
def unet_model(input_size=(128, 128, 3), num_classes=num_classes):
    inputs = Input(input_size)
    
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    
    # Bottleneck
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    # Decoder (Expanding Path)
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Instantiate the model
model = unet_model(input_size=(128, 128, 3))




# Compile Model
model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

# Train Model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=10
)
import matplotlib.pyplot as plt
import numpy as np

# Randomly select an image from X_val
random_index = 4
random_image = X_val[random_index]
random_image = np.expand_dims(random_image / 255.0, axis=0)  # Normalize image

# Predict segmentation mask
predicted_mask = model.predict(random_image)  # Output shape: (1, height, width, num_classes)
predicted_mask = np.argmax(predicted_mask, axis=-1)  # Convert softmax to class indices
predicted_mask = predicted_mask.squeeze()  # Remove batch dimension

# Ground Truth Mask
true_mask = y_val[random_index]  # Shape: (height, width, num_classes)
true_mask = np.argmax(true_mask, axis=-1)  # Convert one-hot to class indices

# Display the original image, true mask, and predicted mask
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(X_val[random_index].squeeze(), cmap='gray')  # Use the original image for display
plt.title("Original Image")
plt.axis('off')

# Ground Truth Mask
plt.subplot(1, 3, 2)
plt.imshow(true_mask, cmap='gray')  # Single-channel mask
plt.title("Ground Truth Mask")
plt.axis('off')

# Predicted Mask
plt.subplot(1, 3, 3)
plt.imshow(predicted_mask, cmap='gray')  # Single-channel mask
plt.title("Predicted Mask")
plt.axis('off')

plt.tight_layout()
plt.show()


