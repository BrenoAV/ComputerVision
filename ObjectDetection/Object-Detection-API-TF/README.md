# Object Detection API - TensorFlow

In this repository I replicated the pipeline for use the Object Detection API. You can find here: [Training Custom Object Detector](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html)

I've used a custom dataset which was made the annotation using the [labelImg](https://github.com/heartexlabs/labelImg). More explanation in the [jupyter notebook](Object_Detection_API.ipynb)

## Errors

After running: `python object_detection/builders/model_builder_tf2_test.py` I got this:

```
ImportError: cannot import name 'builder' from 'google.protobuf.internal' (/home/brenoav/anaconda3/envs/od-api/lib/python3.8/site-packages/google/protobuf/internal/__init__.py)
```

How I solved this problem: https://stackoverflow.com/questions/71759248/importerror-cannot-import-name-builder-from-google-protobuf-internal
