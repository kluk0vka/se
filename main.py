from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

image_path = "kotiki2.jpg"
image = cv2.imread(image_path)

results = model(image)

results[0].save(filename="detected_objects.jpg")

print("детекция завершена, результат сохранен в 'detected_objects.jpg'.")
