import logging
import random
import os
from pathlib import Path
import glob
import shutil

logging.basicConfig(level=logging.INFO)

DATASET_PATH = Path(
    "/mnt/files/Datasets/Traffic-Signs-Dataset-in-YOLO-format/dataset/ts/"
)
TEST_SIZE = 0.2
TRAIN_SIZE = 1 - TEST_SIZE
DATASET_ROOT = Path("datasets/traffic-light/")
TRAIN_IMG_PATH = Path(os.path.join(DATASET_ROOT, "images/train/"))
TRAIN_LABELS_PATH = Path(os.path.join(DATASET_ROOT, "labels/train/"))
VAL_IMG_PATH = Path(os.path.join(DATASET_ROOT, "images/val/"))
VAL_LABELS_PATH = Path(os.path.join(DATASET_ROOT, "labels/val/"))

# TODO: Save the files and adding some logging to veryfing the process
# TODO: Divide into test too

if __name__ == "__main__":
    TRAIN_IMG_PATH.mkdir(parents=True, exist_ok=True)
    TRAIN_LABELS_PATH.mkdir(parents=True, exist_ok=True)
    VAL_IMG_PATH.mkdir(parents=True, exist_ok=True)
    VAL_LABELS_PATH.mkdir(parents=True, exist_ok=True)

    all_images_path = glob.glob(os.path.join(DATASET_PATH, "*.jpg"))
    all_annot_path = glob.glob(os.path.join(DATASET_PATH, "*.txt"))

    all_images_path.sort()
    all_annot_path.sort()

    # Verifying if the the number of images is equals to number of annot
    assert len(all_images_path) == len(all_annot_path)

    # Removing the extension .jpg and .txt
    images_name = [filename.split(".")[0] for filename in all_images_path]
    annot_name = [filename.split(".")[0] for filename in all_annot_path]

    # Verifying if the names are equals (images and annot)
    assert images_name == annot_name

    train_images_name = random.sample(images_name, int(TRAIN_SIZE * len(images_name)))
    val_images_name = set(images_name) - set(train_images_name)

    # Adding the extesion .jpg
    train_images_path = [filename + ".jpg" for filename in train_images_name]
    val_images_path = [filename + ".jpg" for filename in val_images_name]

    # Adding the extesion .txt
    train_annot_path = [filename + ".txt" for filename in train_images_name]
    val_annot_path = [filename + ".txt" for filename in val_images_name]

    # Train
    for img_file, annot_file in zip(train_images_path, train_annot_path):
        shutil.copy(img_file, TRAIN_IMG_PATH)
        shutil.copy(annot_file, TRAIN_LABELS_PATH)

    # Val
    for img_file, annot_file in zip(val_images_path, val_annot_path):
        shutil.copy(img_file, VAL_IMG_PATH)
        shutil.copy(annot_file, VAL_LABELS_PATH)
