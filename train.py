from ultralytics import YOLO
import os


print(os.getcwd() + "/datasets/phenobench.yaml")
# dataset_path = os.path.join(os.getcwd(),  'datasets', 'phenobench.yaml')
# # Load a COCO-pretrained YOLOv8n model
model = YOLO('yolov8n-seg.yaml')  # build a new model from YAML
results = model.train(data=(os.getcwd() + "/datasets/phenobench.yaml"), epochs=100)  # train

