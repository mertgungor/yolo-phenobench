from ultralytics import YOLO 
import cv2
import numpy as np
from PIL import ImageDraw, Image
import os

crop_color = (255, 0,   0,   128)  # Red color with 50% transparency (BGR with alpha)
weed_color = (0,   0,   255, 128)  # Red color with 50% transparency (BGR with alpha)
text_color = (255, 255, 255, 255)  # Red color with 50% transparency (BGR with alpha)

id2color = {0: crop_color, 1: weed_color}

def show_mask(result):
    image = result[0].orig_img
    height, width, _ = image.shape
    print(height, width)

    mask = np.zeros_like(image)

    global crop_color
    global weed_color
    global text_color
    global id2color

    id2class = result[0].names

    for i in range(len(result[0].masks)):

        # print(id2class[result[0].boxes[i].cls.item()])

        pixel_polygons = np.array([(int(polygon[0] * width), int(polygon[1] * height)) for polygon in result[0].masks[i].xyn[0]], dtype=np.int32)
        # box = np.array([(int(box[0] * width), int(box[1] * height)) for box in result[0].boxes[i].xywhn[0]], dtype=np.int32)
        box = result[0].boxes[i].xyxyn[0]

        x1 = int(box[0] * height)
        y1 = int(box[1] * width)

        cv2.rectangle(
            image, 
            (x1, y1), 
            (int(box[2] * height), int(box[3] * width)), 
            id2color[result[0].boxes[i].cls.item()], 
            2)

        cv2.putText(image, id2class[result[0].boxes[i].cls.item()], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

        cv2.fillPoly(mask, [pixel_polygons], id2color[result[0].boxes[i].cls.item()])

    result_image = cv2.addWeighted(image, 1, mask, 1, 0)  # Adjust the alpha value (0.5) for transparency

    # Display the result
    im_show(result_image)


def im_show(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


model = YOLO('yolov8n-seg.pt')  # load an official model
model = YOLO('best.pt')  # load an official model
current_dir = os.getcwd()
path  = current_dir + "/datasets/phenobench/images/train"

for image in os.listdir(path):

    result = model(path + '/' + image)

    try:
        show_mask(result)
    except:
        continue
