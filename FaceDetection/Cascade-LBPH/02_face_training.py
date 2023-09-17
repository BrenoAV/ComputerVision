#!/usr/bin/env python
"""

A script to training the model to be able to recognize faces

Usage:

- Normal usage:

$ python 02_face_training.py --records=records

This will take the records of the folder 'records/' and train the model to recognize
different faces (ids)

"""
import os
import logging
import argparse
from datetime import datetime
import cv2
import numpy as np
from src.cascade_tools import create_cascade_classifier, download_haarcascade
from src.record import Record
from src.dir_tools import create_dir

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(
    description="Script to train the model using the records"
)
parser.add_argument(
    "--records",
    required=True,
    type=str,
    help="Directory where the records are being stored",
)
args = parser.parse_args()


def main():
    """main function"""
    record = Record(record_path=args.records)
    record.load()
    # Download the trained weights to detect faces only
    haarcascade_filepath = download_haarcascade()
    detector = create_cascade_classifier(haarcascade_filepath)
    face_samples = []
    records_ids = []
    if record.dict_records:
        # Go to all the records and take the photos from each record id and
        # accumulate the face detection with the records ids
        for record_id, images_path in record.dict_records.items():
            for image_path in images_path:
                img = cv2.imread(image_path)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(img_gray)
                for x, y, w, h in faces:
                    face_cropped = img_gray[y : y + h, x : x + w]
                    face_samples.append(face_cropped)
                    records_ids.append(record_id)

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(face_samples, np.array(records_ids))
        create_dir("trainer")
        time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # Saving the last time that the models was trained with the dataset
        filepath = os.path.join("trainer", f"trainer-{time}.yml")
        recognizer.write(filepath)
        logging.info(f"Trainer: {filepath} saved!")


if __name__ == "__main__":
    main()
