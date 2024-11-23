# Model Training code 

# from ultralytics import YOLO
# model = YOLO(r'C:\\Users\\Omkar\Desktop\\CVDL Dataset-20241122T193125Z-001\\CVDL Dataset\\object detection\\yolov8n.pt')
# model.train(data=r'C:\\Users\\Omkar\Desktop\\CVDL Dataset-20241122T193125Z-001\\CVDL Dataset\\object detection\\Persian_Car_Plates_YOLOV8\data.yaml', epochs=1)


# Detection  code  using the best.pt

from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# importing the train model 
model = YOLO("runs\detect\\train\weights\\best.pt")

def display_images(original_img_rgb, annotated_img_rgb):
   plt.figure(figsize=(12, 6))

   plt.subplot(1, 2, 1)
   plt.imshow(original_img_rgb)
   plt.title('Original Image')
   plt.axis('off')

   plt.subplot(1, 2, 2)
   plt.imshow(annotated_img_rgb)
   plt.title('Detected Objects')
   plt.axis('off')

   plt.tight_layout()   
   plt.show()

path = "object detection\Persian_Car_Plates_YOLOV8\\valid\images\\203_png.rf.5507fa5f6fbe956191d2f57f36cd0005.jpg"
image = cv2.imread(path)
results = model(image)
result = results[0]

for box in result.boxes.xyxy:
    x1, y1, x2, y2 = map(int, box[:4])
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

annoted_img = result.plot()
annotated_img_rgb = cv2.cvtColor(annoted_img, cv2.COLOR_BGR2RGB)
display_images(image,annotated_img_rgb)




