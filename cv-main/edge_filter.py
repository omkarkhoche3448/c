import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# path = input("Enter the image path")
path = 'edge_detection_images\\7d6c1119-00000000.jpg'
image = cv2.imread(path)

def plot(result,rtitle):
    plt.figure(figsize = (6,6))
    plt.subplot(1,2,1)
    plt.title("Original image")
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.title(rtitle)
    plt.imshow(result)
    plt.show()

choice = 0
while True:
    print("1.Median filters")   
    print("2.Box filters")
    print("3.Guassian filters")
    # Now edge detectors
    print("4.Sobel filter")
    print("5.canny edge")     
    print("6.prewitt operator")
    print("7. Exit")

    choice = int(input("Enter your choice - "))

    if choice == 1:
        result = cv2.medianBlur(image, ksize=7)
        plot(result,"Median")
    
    if choice == 2:
        result = cv2.boxFilter(image, -1 ,ksize=(5,5))
        # result = cv2.blur(image, ksize=(5, 5))  same apply average filter
        plot(result,"box")
    
    if choice == 3:
        sigma = 1 # take from user
        result = cv2.GaussianBlur(image, (3,3), sigma)
        plot(result,"guassian")
    
    if choice == 4:
        sobel_x = cv2.Sobel(image, -1, 1, 0, ksize=5) 
        sobel_y = cv2.Sobel(image, -1, 0, 1, ksize=5) 
        # sobel_combined = cv2.magnitude(sobel_x, sobel_y) # all edge
        plot(sobel_x,"sobel")
    
    if choice == 5:
        result = cv2.Canny(image, threshold1=100, threshold2=200)
        plot(result,"canny")
    
    if choice == 6:
        kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        prewitt_x = cv2.filter2D(image, ddepth=-1, kernel=kernel_x)  # Horizontal edges
        prewitt_y = cv2.filter2D(image, ddepth=-1, kernel=kernel_y)
        plot(prewitt_x,'x')
        plot(prewitt_y,'y')
    
    if choice == 7:
        exit








