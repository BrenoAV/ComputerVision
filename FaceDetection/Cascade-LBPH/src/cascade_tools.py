import logging
import os
import requests

import cv2

from src.dir_tools import create_dir

URL_HAARCASCADES = "https://raw.githubusercontent.com/kipr/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"


def download_haarcascade() -> str:
    response = requests.get(URL_HAARCASCADES, timeout=20)
    logging.info(f"Downloading from {URL_HAARCASCADES}...")
    create_dir("haarcascades")
    haarcascade_filepath = os.path.join(
        "haarcascades", "haarcascade_frontalface_default.xml"
    )
    with open(haarcascade_filepath, "wb") as file:
        file.write(response.content)
    logging.info(f"{haarcascade_filepath} saved!")
    return haarcascade_filepath


def create_cascade_classifier(haarcascade_filepath: str) -> cv2.CascadeClassifier:
    detector = cv2.CascadeClassifier(haarcascade_filepath)
    return detector
