import os
import numpy as np
import os, json, cv2
import pickle

category_to_id = {
    "crop": 0,
    "weed": 1,
}

def get_weed_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    # print(list(imgs_anns.values())[0])

    dataset_dicts = []

    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        # print(filename)
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():

            region_attributes = anno["region_attributes"]
            # print(region_attributes)
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            center_x = anno["center_x"]
            center_y = anno["center_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            if(len(px)==4):
                print("=========================================================")
                print(px)
                print("=========================================================")

            if region_attributes:
                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "segmentation": [poly],
                    "category_id": category_to_id[region_attributes["name"]],
                    "keypoints": [center_x, center_y, 1],
                    
                }

                objs.append(obj)
        record["annotations"] = objs
        
        dataset_dicts.append(record)
    return dataset_dicts

data_dicts = get_weed_dicts("/home/mert/JSON2YOLO/datasets/phenobench/images/train/")
with open('phenobench.pickle', 'wb') as file:
    pickle.dump(data_dicts, file)

for data_dict in data_dicts:
    print(data_dict.keys())
    break