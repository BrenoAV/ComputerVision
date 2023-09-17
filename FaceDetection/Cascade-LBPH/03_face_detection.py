"""
A script to detecting in real-time face

Usage:

- Normal usage:

$ python 03_face_detection.py --records="records" \
    --detector_filepath="haarcascades/haarcascade_frontalface_default.xml" \
    --recognizer_filepath="trainer/trainer-YYYY-MM-DD-HH-MM-SS.yml"

This will take the records (map the id with a name), detector of face (Cascade), 
and the trained recognizer(LBPH). Open a window that will detect faces and recognize.


"""
import argparse
import logging
import cv2

from src.cascade_tools import create_cascade_classifier
from src.record import Record

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
parser.add_argument(
    "--detector_filepath",
    required=True,
    type=str,
    help="Filepath which haarcascade (.xml) you'll use it",
)
parser.add_argument(
    "--recognizer_filepath",
    required=True,
    type=str,
    help="Filepath which recognizer (.yml) you'll use it",
)
args = parser.parse_args()


def main():
    """main function"""

    record = Record(record_path=args.records)
    record.load()

    detector = create_cascade_classifier(args.detector_filepath)

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.read(args.recognizer_filepath)

    logging.info("Preparing the camera... ")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    logging.info("Camera working!")

    # Define min window size to be recognized as a face
    minW = int(0.1 * cap.get(3))
    minH = int(0.1 * cap.get(4))

    names = record.dict_id_name

    while True:
        ret, img = cap.read()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(
            gray_img, scaleFactor=1.2, minNeighbors=5, minSize=(minW, minH)
        )

        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + w), (0, 255, 0), 2)
            rec_id, conf = recognizer.predict(gray_img[y : y + h, x : x + w])

            if conf < 50:
                name_id = names[str(rec_id)]
                conf = "  {0}%".format(round(100 - conf))
            else:
                name_id = "unknown"
                conf = "  {}%".format(round(100 - conf))
            cv2.putText(
                img,
                str(name_id),
                (x + 5, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                img,
                str(conf),
                (x + 5, y + h - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                1,
            )

        cv2.imshow("Detecting Face", img)
        k = cv2.waitKey(10) & 0xFF
        if k == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
