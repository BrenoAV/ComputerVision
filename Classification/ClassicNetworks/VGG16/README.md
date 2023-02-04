# VGG

- **Framework:** TensorFlow
- **Dataset:** Cat x Dog (https://www.kaggle.com/datasets/chetankv/dogs-cats-images)
- **Paper Link:** https://paperswithcode.com/model/vgg
- I'm taking my handwritten notes, and I'll not explain much.
- I didn't use transfer learning.
- I didn't use the dataset "2012 ILSVRC" because is a huge dataset. I'm using the [Cat-vs-Dogs dataset](https://www.kaggle.com/datasets/chetankv/dogs-cats-images) available on Kaggle. The others architectures I'll search for others datasets.
- I'm using VGG-A [(pg 3)](https://arxiv.org/abs/1409.1556)
- Because I'm using a small dataset than VGG was prior made, I've changed some hyperparameters:
  - FC1 4096 neurons &rarr; 216 neurons
  - FC2 4096 neurons &rarr; 64 neurons
  - Output 1000 neurons (multi-class classification) &rarr; 2 neurons (binary classification)
- For this example as you can see in the [jupyter](VGG.ipynb) in 90 epochs the learning rate doesn't change of 0.01, maybe using other optimizer like Adam or changing the initial learning rate we could reach better accuracy value than 0.8810. The graph shows us that the accuracy hast not stabilized, because the value didn't decrease.
