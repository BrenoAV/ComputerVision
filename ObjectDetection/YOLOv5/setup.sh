#! /usr/bin/bash

DIR_YOLOV5="yolov5"
GITHUB_LINK="https://github.com/ultralytics/yolov5"
WEIGHTS_PRETRAINED="https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt"

if [[ -d $DIR_YOLOV5 ]] then
    echo "Directory $DIR_YOLOV5 already exists in your path"
else
    git clone $GITHUB_LINK # Clonning
fi

cd yolov5

echo "Copying the scripts/ folder to yolov5/"
cp -r ../scripts/ ./
echo "Copying the dataset.yaml to yolov5/"
cp ../dataset.yaml ./
echo "Creating datasets/ folder"
mkdir -p datasets/
echo "Downloading the weights pretrained from $WEIGHTS_PRETRAINED ..."
wget $WEIGHTS_PRETRAINED
echo "Finished"
