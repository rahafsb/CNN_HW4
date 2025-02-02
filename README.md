# CNN-Based Flower Classification

## Overview
This project focuses on utilizing Convolutional Neural Networks (CNNs) with Transfer Learning to classify flower images into their corresponding categories. We employ two pre-trained models, VGG19 and YOLOv5, and adapt them for classification tasks. The model predicts the probability of an image belonging to each category.

## Dataset
The primary dataset used is the **Oxford 102 Category Flower Dataset** available at:
[https://www.robots.ox.ac.uk/~vgg/data/flowers/102/](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

Additional images from other repositories can be incorporated to improve accuracy.

## Project Requirements
- The code must be implemented in **Python**.
- Deep learning frameworks: **Keras, TensorFlow, or PyTorch**.
- **Pre-trained models:** VGG19 and YOLOv5.
- The dataset should be **randomly split** into:
  - **Training set (50%)**
  - **Validation set (25%)** (for hyperparameter tuning)
  - **Test set (25%)**
- The random split should be repeated at least **twice** to ensure robustness.
- The classification model should achieved **at least 70% test accuracy** with at least one of the models.

## Preprocessing Steps
1. **Image Resizing:** Standardize all images to a fixed input size compatible with VGG19 and YOLOv5.
2. **Normalization:** Scale pixel values to [0,1] or standardize using mean subtraction.


## Model Details
### VGG19-based Model
https://github.com/rahafsb/CNN_HW4/blob/main/VGG19.ipynb
- Utilize **VGG19** as a feature extractor.
- Remove the final classification layer and replace it with:
  - Fully Connected (Dense) Layers
  - Dropout for regularization
  - Softmax activation for multi-class classification

### YOLOv5-based Model
https://github.com/rahafsb/CNN_HW4/blob/main/yolov5.ipynb
- Convert YOLOv5 from object detection to classification mode.
- Extract features from its backbone network.
- Add custom **fully connected layers** for classification.
- Fine-tune selected layers for improved performance.

## Training Procedure
- Use **Cross-Entropy Loss** as the loss function.
- **Optimizer:** Adam or SGD with learning rate tuning.
- Implement **early stopping** and learning rate scheduling.
- Train both models for multiple epochs and compare performance.

## Evaluation
- Plot **Accuracy vs. Epochs** for training and validation sets.
- Plot **Cross-Entropy Loss vs. Epochs** for training, validation, and test sets.
- Report classification accuracy on the **test set**.

## Outcome
VGG19:
-	Cycle 1 Test Accuracy: 80.18%
- Cycle 2 Test Accuracy: 81.20%
-	Average Test Accuracy: 80.69%



## References
- Oxford 102 Flowers Dataset
- VGG19: Simonyan & Zisserman, 2014
- YOLOv5: Ultralytics, 2020

---
This project explores CNN-based classification using pre-trained models and transfer learning to achieve robust flower recognition.
https://github.com/rahafsb/CNN_HW4/tree/main

