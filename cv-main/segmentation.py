import cv2
import numpy as np
import matplotlib.pyplot as plt

# Image path
img_path = 'edge_detection_images\\7d6c1119-00000000.jpg'

def show_images(original, processed, title_processed):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(processed, cmap='gray')
    plt.title(title_processed)
    plt.axis('off')
    plt.show()

# Load and resize the image
image = cv2.imread(img_path)
if image is None:
    raise ValueError("Image not found. Please check the path.")
image = cv2.resize(image, (250, 250))
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

while True:
    print("\nSelect a segmentation method:")
    print("1. Binary Thresholding")
    print("2. Otsu's Thresholding")
    print("3. Adaptive Thresholding")
    print("4. Region Growing (Scratch Method)")
    print("5. Region Growing (Flood Fill Method)")
    print("6. Watershed Segmentation")
    print("7. Exit")
    
    choice = input("Enter your choice (1-7): ")
    
    if choice == '1':
        # Binary Thresholding
        _, result = cv2.threshold(image_gray, 120, 255, cv2.THRESH_BINARY)
        show_images(image_gray, result, "Binary Thresholding")
    
    elif choice == '2':
        # Otsu's Thresholding
        _, result = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        show_images(image_gray, result, "Otsu's Thresholding")
    
    elif choice == '3':
        # Adaptive Thresholding
        result = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, blockSize=11, C=2)
        show_images(image_gray, result, "Adaptive Thresholding")
    
    elif choice == '4':
        # Region Growing (Scratch Method)
        height, width = image_gray.shape
        middle_x, middle_y = height // 2, width // 2
        print(f"Suggested seed coordinates: ({middle_x}, {middle_y})")
        seed_x = int(input("Enter seed point x-coordinate: "))
        seed_y = int(input("Enter seed point y-coordinate: "))
        
        segmented = np.zeros_like(image_gray)
        region = {(seed_x, seed_y)}

        while region:
            x, y = region.pop()
            if segmented[x, y] == 0:
                segmented[x, y] = 255

                # Check neighbors
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if 0 <= x + i < height and 0 <= y + j < width:
                            if abs(int(image_gray[x, y]) - int(image_gray[x + i, y + j])) < 15:
                                region.add((x + i, y + j))

        show_images(image_gray, segmented, "Region Growing (Scratch Method)")
    
    elif choice == '5':
        # Region Growing (Flood Fill Method)
        height, width = image_gray.shape
        middle_x, middle_y = height // 2, width // 2
        print(f"Suggested seed coordinates: ({middle_x}, {middle_y})")
        seed_x = int(input("Enter seed point x-coordinate: "))
        seed_y = int(input("Enter seed point y-coordinate: "))
        
        segmented = np.zeros_like(image_gray)
        cv2.floodFill(segmented, None, (seed_x, seed_y), 255)
        
        show_images(image_gray, segmented, "Region Growing (Flood Fill Method)")
    
    elif choice == '6':
        # Watershed Segmentation
        ret, binary_thresh = cv2.threshold(image_gray, 120, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        binary_thresh = cv2.morphologyEx(binary_thresh, cv2.MORPH_CLOSE, kernel)
        
        dist_transform = cv2.distanceTransform(binary_thresh, cv2.DIST_L2, 5)
        _, markers = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        markers = markers.astype(np.int32)
        markers = cv2.watershed(image, markers)
        image[markers == -1] = [255, 0, 0]  # Mark boundaries in red
        
        show_images(image_gray, image, "Watershed Segmentation")
    
    elif choice == '7':
        print("Exiting the program.")
        break
    
    else:
        print("Invalid choice! Please select 1-7.")