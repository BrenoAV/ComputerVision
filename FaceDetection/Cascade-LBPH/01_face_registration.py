#!/usr/bin/env python
"""
Script to store a face record separately. An import file will be created called id-name.json
that will be responsible for persisting the data that maps the ID to a name.

Usage:

- Show info about the arguments

$ python 01_face_registration.py --help

- Normal usage:

$ python 01_face_registration.py --records="records" --id=1 --name="my_name"

This will create a folder if records doesn't exits and create a folder 001 and storage
your pictures

"""
import argparse
import logging
import time
import cv2
from src.record import Record

logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser(description="Script to create a record of the face")
parser.add_argument(
    "--records",
    required=True,
    type=str,
    help="Directory where the records are being stored",
)
parser.add_argument(
    "--id", required=True, type=int, help="ID to be storage on the records"
)
parser.add_argument(
    "--name", required=True, type=str, help="A name associated with the ID"
)
args = parser.parse_args()


def main():
    """main function"""
    record = Record(record_path=args.records)
    record.load()

    if args.id in record.dict_records:
        raise SystemExit(
            f"The {args.id} already exists in the records! Please, try another one..."
        )

    logging.info("Preparing the camera... ")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    logging.info("Camera working!")

    logging.info("Press 't' to take a photo to be recorded on the records...")
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Face Registration", frame)
            k = cv2.waitKey(20) & 0xFF
            if k == ord("q"):
                break
            if k == ord("t"):
                record.save_record(frame, str(args.id), args.name)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
