# Dual-Stream CNN for CIFAR-10 Image Classification

## Project Overview
This project implements a novel dual-stream Convolutional Neural Network (CNN) architecture for classifying the CIFAR-10 dataset. The approach is inspired by the human visual system, which processes luminance and chrominance information separately. By decomposing RGB images into YUV color space and processing these channels through separate streams, this architecture achieves significantly better performance than traditional machine learning methods (such as SVC, Random Forest, and Logistic Regression).

## Dataset
The CIFAR-10 dataset consists of 60,000 32×32 color images across 10 different categories:
- Airplanes
- Automobiles
- Birds
- Cats
- Deer
- Dogs
- Frogs
- Horses
- Ships
- Trucks

The dataset is split into 50,000 training images and 10,000 test images, with a balanced number of samples in each class.

## Methodology

### Data Preprocessing and Augmentation
- Images are normalized to the [0,1] range
- One-hot encoding is applied to the labels
- Dataset is split with 80% training and 20% validation ratio
- Comprehensive data augmentation strategy includes:
  - Random rotation (±15 degrees)
  - Random horizontal and vertical shifts (up to 10%)
  - Random horizontal flips
  - Random zoom (up to 10%)
- This augmentation strategy effectively quadruples the original training set

### Model Architecture
The dual-stream CNN architecture works as follows:

1. Input RGB images (32×32×3) are converted to YUV color space using a Lambda layer
2. The Y channel (luminance) is processed through one dedicated stream, focusing on shape and texture
3. The UV channels (chrominance) are processed through a parallel stream, focusing on color information
4. Both streams have identical layer structures:
   - Convolutional layers with 3×3 kernels
   - Batch normalization
   - ReLU activations
   - Dropout (increasing rates with depth, from 0.2 to 0.3)
   - Max pooling layers
5. Features from both streams are concatenated after independent processing
6. The merged features are further processed through additional convolutional blocks
7. Final classification uses fully connected layers with softmax activation

### Training Setup
- Optimizer: Adam
- Loss function: Categorical cross-entropy
- Batch size: 64
- Maximum epochs: 50
- Callbacks:
  - ModelCheckpoint to save the best model
  - EarlyStopping with patience of 10 to prevent overfitting
  - ReduceLROnPlateau to adaptively adjust the learning rate (reducing by 20% when validation loss shows no improvement for 5 epochs)

## Results
- The dual-stream CNN achieved approximately 90% accuracy on the test set
- This significantly outperforms traditional machine learning methods from previous work:
  - SVC with PCA: 57.63% accuracy
  - Random Forest: 40.98% accuracy
  - Logistic Regression: 33.96% accuracy
- Training time was reduced by 73.8% compared to SVC (850 seconds vs. 3251 seconds)
- Performance varies by class:
  - Best performing classes: automobile (95%), frog (93%), horse (93%), ship (94%), and truck (94%)
  - Most challenging classes: cat (78%) and dog (84%)

## Conclusion
The dual-stream CNN architecture effectively solves the problems of low accuracy and long training times faced by traditional machine learning methods for image classification. The separation of luminance and chrominance processing channels proves to be an effective strategy, inspired by human visual processing. This approach not only delivers superior performance but also demonstrates significant advantages in training efficiency, interpretability, and resource utilization.

## How to Run the Code

1. Install the required libraries:
 Install the required libraries:
pip install tensorflow numpy matplotlib seaborn scikit-learn

2. Run the Jupyter notebook:
jupyter notebook Project.ipynb

3. Follow the steps in the notebook to:
   - Load and preprocess the CIFAR-10 dataset
   - Build the dual-stream CNN model
   - Train the model
   - Evaluate performance and visualize results

## Requirements
- Python 3.8
- TensorFlow
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Future Work
- Explore more flexible inter-stream interaction mechanisms
- Apply the architecture to more complex visual tasks
- Develop lighter and more efficient model variants
- Introduce residual connections
- Explore more diverse color space decompositions
- Implement more advanced regularization and data augmentation techniques
- Apply transfer learning approaches