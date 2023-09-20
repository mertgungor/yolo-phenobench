from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO('yolov8n-seg.yaml')  # build a new model from YAML
results = model.train(data='/home/mert/JSON2YOLO/datasets/phenobench.yaml', epochs=100)  # train

