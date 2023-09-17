import json
from typing import Dict, Optional
import logging
import os
import uuid
from pathlib import Path
import cv2
from numpy.typing import NDArray
from src.dir_tools import create_dir


class Record:
    def __init__(self, record_path: str) -> None:
        self.record_path = record_path
        self.record_path_obj = Path(record_path)
        create_dir(record_path)
        self.dict_records: Optional[Dict[int, list]] = None
        self.dict_id_name: Optional[Dict[str, str]] = None
        self.ids: list[int] = []
        self.num_records: int = 0

    def load(self) -> None:
        # load the records from the specified directory
        if self.record_path_obj.exists():
            self.dict_records = {
                int(i): [
                    os.path.join(os.path.join(self.record_path, i), photo)
                    for photo in os.listdir(os.path.join(self.record_path, i))
                ]
                for i in os.listdir(path=self.record_path_obj)
                if not i.endswith(".json")
            }
        else:
            create_dir(self.record_path)
        # open the json file with the id - name if exists
        if Path(os.path.join(self.record_path, "id-name.json")).exists():
            with open(
                os.path.join(self.record_path, "id-name.json"),
                "r",
                encoding="utf-8",
            ) as json_file:
                self.dict_id_name = json.load(json_file)
        else:
            self.dict_id_name = {}
        if self.dict_records:
            self.num_records = len(self.dict_records.keys())
            self.ids = list(self.dict_records.keys())

    def save_record(self, img: NDArray, record_id: str, name: str) -> None:
        """Save a record (gray image) in a directory (record id) and refresh records paths

        Args:
            img (NDArray): an image (BGR) to be saved
            record_id (int): the id of the record
            name (str): the name associated with the id
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        id_record_path = os.path.join(self.record_path, str(record_id).rjust(3, "0"))
        create_dir(id_record_path)
        record_filename = os.path.join(id_record_path, f"{uuid.uuid4()}.png")
        cv2.imwrite(record_filename, img=img_gray)
        # Append a new id - name on the json file
        if self.dict_id_name is not None:
            with open(
                os.path.join(self.record_path, "id-name.json"), "w", encoding="utf-8"
            ) as json_file:
                self.dict_id_name[record_id] = name
                json.dump(self.dict_id_name, json_file, indent=2, ensure_ascii=True)
        logging.info(f"Image '{record_filename}' saved!")
        self.load()

    def __repr__(self):
        return f"Record Path: {self.record_path}\n"
