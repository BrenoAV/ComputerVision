import xml.etree.ElementTree as ET
from typing import Tuple, List
import cv2
import PIL.Image as Image
import numpy as np
from sklearn.datasets import load_files


def ssearch(img):
    """Selective Search using Fast approach
    
    Args:
        img: numpy array representation of the image
    Returns:
        results: bbouxes with possible objects (-1, x, y, w, h)
    
    """
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    results = ss.process()
    return results

def save_roi(img_roi, filepath, size:Tuple[int]):
    """Function to save the roi images after crop and resize the image to feed the base model
    
    Args:
        img_roi: numpy array (represeation of the image)
        filepath: Str - filepath to save the image
        size: Tuple with the image size
    
    """
    img_roi = cv2.resize(img_roi, size)
    img_roi = Image.fromarray(img_roi)
    img_roi.save(filepath)
    
def read_annot(xml_filepath: str):
    """Read PASCAL annotation to get the filename and bouding boxes
    
    Args:
        xml_filepath (str): .xml file annotation
    Returns:
        filename (str): filename associated with the annotation
        list_with_all_boxes (List): all boxes annotated in the image
    
    """
    tree = ET.parse(xml_filepath)
    root = tree.getroot()

    list_with_all_boxes = []

    filename = root.find("filename").text

    for boxes in root.iter("object"):

        for box in boxes.findall("bndbox"):
            x_min, y_min, x_max, y_max = None, None, None, None
            x_min = int(box.find("xmin").text)
            y_min = int(box.find("ymin").text)
            x_max = int(box.find("xmax").text)
            y_max = int(box.find("ymax").text)

            bbox = [x_min, y_min, x_max, y_max]
            list_with_all_boxes.append(bbox)

    return filename, list_with_all_boxes

def get_iou(bbox1: List[int], bbox2: List[int]):
    """Compute the IOU between two bounding box. If there's no
    intersection return zero.

    Args:
        bbox1: List - [x1, y1, x2, y2]
        bbox2: List - [x1, y1, x2, y2]

    Returns:
        Return IoU between the bbox1 and bbox2.


    """
    assert bbox1[0] < bbox1[2]
    assert bbox1[1] < bbox1[3]
    assert bbox2[0] < bbox2[2]
    assert bbox2[1] < bbox2[3]

    x_left = max(bbox1[0], bbox2[0])
    x_right = min(bbox1[2], bbox2[2])
    y_top = max(bbox1[1], bbox2[1])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)

    bbox1_area = (bbox1[3] - bbox1[1]) * (bbox1[2] - bbox1[0])
    bbox2_area = (bbox2[3] - bbox2[1]) * (bbox2[2] - bbox2[0])

    iou = intersection / (bbox1_area + bbox2_area - intersection)

    assert iou <= 1.0
    assert iou >= 0.0

    return iou

def nms(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []
  
    bboxes = np.array(bounding_boxes)
    
    scores = np.array(confidence_score)
    
    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    
    while bboxes.size > 0:
        order = np.argsort(scores)
        high_idx = order[-1]

        picked_boxes.append(bboxes[high_idx])
        picked_score.append(scores[high_idx])

        iou_list = np.array([get_iou(bboxes[high_idx], bbox) for bbox in bboxes])

        # Here we remove the IoU between bboxes[high_idx] and bboxes[high_idx] = 1
        left_idx = np.where(iou_list < threshold)  

        bboxes = bboxes[left_idx]
        scores = scores[left_idx]
            
    return picked_boxes, picked_score

def load_dataset(data_path, background_limit=4000):
    data = load_files(data_path)
    filenames = data['filenames']
    targets = data['target']
    target_names = data['target_names']
    
    negative_count = 0
    max_negative = background_limit
    X = []
    y = []
    for i, filename in enumerate(filenames):
        if not (targets[i] == 0 and negative_count > max_negative):
            if targets[i] == 0:
                negative_count += 1
            img = cv2.imread(filename)
            X.append(img)
            y.append(targets[i])

    X = np.array(X)
    y = np.array(y)
    return X, y