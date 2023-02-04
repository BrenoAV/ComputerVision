# AlexNet

- **Framework:** TensorFlow
- **Dataset:** Cats-Dogs (https://www.kaggle.com/datasets/chetankv/dogs-cats-images)
- **Paper Link:** https://paperswithcode.com/model/alexnet
- I'm taking my handwritten notes, and I'll not explain much.
- I didn't use transfer learning.
- I didn't use the dataset "2012 ILSVRC" because is a huge dataset. I'm using the [Cat-vs-Dogs dataset](https://www.kaggle.com/datasets/chetankv/dogs-cats-images) available on Kaggle. The others architectures I'll search for others datasets.
- Because I'm using a small dataset than AlexNet was prior made, I've changed some hyperparameters:
  - FC1 4096 neurons &rarr; 216 neurons
  - FC2 4096 neurons &rarr; 64 neurons
  - Output 1000 neurons (multi-class classification) &rarr; 2 neurons (binary classification)
- For this example as you can see in the [jupyter](AlexNet.ipynb) the model is in overfitting, maybe using some data aug we can reach a better value for accuracy (0.8725)
