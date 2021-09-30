import json
from typing import List

from tqdm import tqdm


def convert_coordinate(size: tuple, box: List) -> tuple:
    """Convert coordinate from xywh to
        x_center, y_center, width, height

    Args:
        size (tuple): size of image
        box (List): box coordinate

    Returns:
        tuple: new yolo coordinate
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    w = box[2]
    h = box[3]
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    x = round(x * dw, 6)
    w = round(w * dw, 6)
    y = round(y * dh, 6)
    h = round(h * dh, 6)
    return (x, y, w, h)


def convert(json_path: str = None, save_path: str = None, classes: List[str] = None):
    """Function to convert Coco json to yolo txt format

    Args:
        json_path (str, optional): Path to the coco json file. Defaults to None.
        save_path (str, optional): The folder which save txt file. Defaults to None.
        classes (List[str], optional): List name of classes. Defaults to None.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    for idx in tqdm(range(len(data["images"]))):
        item = data["images"][idx]
        image_id = item["id"]
        file_name = item["file_name"]
        width = item["width"]
        height = item["height"]
        value = filter(lambda item1: item1["image_id"] == image_id, data["annotations"])
        outfile = open(save_path + "/%s.txt" % (file_name[:-4]), "a+")
        for item2 in value:
            category_id = item2["category_id"]
            value1 = list(
                filter(lambda item3: item3["id"] == category_id, data["categories"])
            )
            name = value1[0]["name"]
            class_id = classes.index(name)
            box = item2["bbox"]
            bb = convert_coordinate((width, height), box)
            outfile.write(str(class_id) + " " + " ".join([str(a) for a in bb]) + "\n")
        outfile.close()
