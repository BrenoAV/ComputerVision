# ResNet (2015)

- **Framework:** TensorFlow
- **Dataset:** Cats-Dogs (https://www.kaggle.com/datasets/chetankv/dogs-cats-images)
- **Paper Link:** https://paperswithcode.com/model/resnet

![ResNet Architecture](src/resnet18.png)

- They introduced a novel `residual module` architecture with "skip connections", this connection allows the gradient to be directly backpropagated to early layers and avoid the vanishing gradient problem. In addition, it allows the model to learn an identity function, which ensures that the layer will perform at least as well as the previous layers.

```mermaid
    graph LR;

    A[x] --> B[Conv];
    B --> C[ReLU];
    C --> D[Conv];
    D --> |"main path = f(x)"|E(("+"));
    A --> |"shortcut path = x"|E;
    E --> |"Add both paths = f(x) + x"|F[ReLU]
    F --> G["relu(f(x) + x)"]
```

- We sum up before the activation function (ReLU)
- The combination of these sum is called `residual block` and we have the following pattern:

```mermaid
    graph BT;
    A{{Input}} --> B[CONV];
    B --> C[POOL];
    C --> D[Residual Block];
    D --> E[Residual Block];
    E --> F[Residual Block];
    F --> G[POOL];
    G --> H[FC];
    H --> I[Softmax]
```

- As we can see the first part is the feature extraction. They used Conv + Pool followed by residual blocks becoming very deep neural networks. After the feature extraction they used fully connected layers for the classifier

## Residual Block

```mermaid
    graph LR;
    A[X] --> B["Conv2D<br>(1x1)"];
    B --> C["BN"];
    C --> D["ReLU"];
    D --> E["Conv2D<br>(3x3)"];
    E --> F["BN"];
    F --> G["ReLU"];
    G --> H["Conv2D<br>(1x1)"];
    H --> I["BN"];
    A --> J["Conv2D<br>(1x1)"];
    J --> K["BN"];
    I --> |"main path"|L(("+"));
    K --> |"shortcut path"|L;
    L --> M["ReLU"];
```

- The main path uses 3 conv layers of sizes: 1x1, 3x3, and 1x1 followed by batch normalization
- The shortcut path can be:
  - Regular shortcut - add the input dimensions to the main path
  - Reduce shortcut - add a conv layer in the shorcut path before merging with the main path (in this case we set up a stride parameter)

- In residual block doesn't have pooling layers to decrease the shape (height, weight), instead of we use conv with stride to decrease.