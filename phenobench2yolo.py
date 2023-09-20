import pickle
import cv2
import numpy as np

label_path = "/home/mert/JSON2YOLO/datasets/phenobench/labels/train/"
image_count = 0

with open('phenobench.pickle', 'rb') as f:
    data = pickle.load(f)

for image in data:

    file_name = image["file_name"].split("/")[-1].split(".")[0] + ".txt"
    # print(image["file_name"])
    # im = cv2.imread(image["file_name"])


    result = ""
    image_count += 1

    for ann in image["annotations"]:

        segmentations = np.array(ann["segmentation"][0])
        result += str(ann["category_id"]) + " " + " ".join(map(str, segmentations/1024)) + "\n"
    
    print(label_path + file_name)

    with open(label_path + file_name, "w") as f:
        f.write(result)

    # if image_count == 10:
    #     break
        
