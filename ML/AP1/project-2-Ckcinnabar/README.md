# Kuan-Chen Project 2
# CIFAR-10 Image Classification with Dimensionality Reduction

## Project Overview
This project analyzes the CIFAR-10 dataset using dimensionality reduction techniques (PCA, t-SNE, UMAP) to improve image classification performance. The analysis compares Support Vector Classifier (SVC), Random Forest, and Logistic Regression models before and after applying these techniques, evaluating their performance, computational efficiency, and visualization capabilities.

## Tasks & Findings

### Task 1: Exploratory Data Analysis
- CIFAR-10 contains 60,000 32×32 RGB images across 10 classes with balanced distribution
- Verified class distribution (5,000 per class in training set)
- Analyzed RGB channel statistics across different classes

### Task 2: Data Visualizations
- Created RGB and individual color channel visualizations
- Analyzed grayscale representations to understand pixel distributions
- Generated channel-specific visualizations to observe RGB component contributions

### Task 3: Preprocessing Steps
- Reshaped 32×32×3 images into 3,072-dimensional vectors
- Applied StandardScaler to normalize pixel values
- Used 500 images per class (5,000 total) for computational efficiency
- Created transformation pipelines for consistent preprocessing

### Task 4: Model Hyperparameters & Performance
**Initial Classification:**
- SVC: kernel='rbf', C=10, gamma='scale' - CV Accuracy: 0.4350
- Random Forest: n_estimators=200, max_depth=None - CV Accuracy: 0.4098
- Logistic Regression: penalty='l2', solver='saga', C=0.1 - CV Accuracy: 0.3396

### Task 5: Dimensionality Reduction Insights
- **PCA Analysis:**
  - ~350-400 components needed for 90% variance retention
  - Reconstruction error decreased significantly with 100-200 components
  - Recognizable reconstructions achieved at 300-500 components
  
- **t-SNE & UMAP Visualization:**
  - 2D projections revealed distinct class clusters
  - t-SNE better preserved local structures while UMAP maintained global relationships
  
- **Performance with Dimensionality Reduction:**
  - PCA+SVC (500 components): 0.6404 CV accuracy
  - PCA+Random Forest (100 components): 0.4656 CV accuracy
  - PCA+Logistic Regression (100 components): 0.4180 CV accuracy
  - t-SNE+SVC and UMAP+SVC provided competitive classification performance

### Task 6: Model Comparison
- SVC with RBF kernel consistently outperformed other models, especially after PCA
- PCA+SVC achieved highest accuracy among all combinations (0.6404)
- Dimensionality reduction significantly reduced computational requirements while maintaining performance
- Non-linear models substantially outperformed linear approaches for image classification

### Task 7: Result Interpretation
- Kernel-based methods are well-suited for image classification after dimensionality reduction
- Dimensionality reduction to 100-500 components provides good balance between efficiency and accuracy
- Performance limited by using reduced training set (5,000 vs 50,000 images)
- Findings suggest computer vision systems should implement non-linear algorithms capable of capturing complex spatial relationships

## Dataset Description
- 60,000 32×32 color images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- 50,000 training images (5,000 per class)
- 10,000 test images (1,000 per class)
- RGB color images (3,072 dimensions per flattened image)

## Repository Structure

- `train.ipynb`: Data exploration, preprocessing, model training, and dimensionality reduction
- `test.ipynb`: Model evaluation and performance comparison
- Trained models:
  - `model_svc.joblib`: Support Vector Classifier 
  - `model_RanF.joblib`: Random Forest Classifier
  - `model_LogR.joblib`: Logistic Regression
  - `model_pca_svc.joblib`: PCA + SVC
  - `model_pca_RanF.joblib`: PCA + Random Forest
  - `model_pca_LogR.joblib`: PCA + Logistic Regression
  - `model_tsne_svc.joblib`: t-SNE + SVC
  - `model_umap_svc.joblib`: UMAP + SVC

## Usage Instructions

### Training
1. Run `train.ipynb` to:
   - Explore and preprocess the CIFAR-10 dataset
   - Train classification models on full-dimensional data
   - Analyze PCA variance and reconstruction quality
   - Implement dimensionality reduction techniques (PCA, t-SNE, UMAP)
   - Train models on reduced feature space and save them

### Testing
1. Run `test.ipynb` to:
   - Load all trained models
   - Evaluate performance on test data
   - Generate confusion matrices
   - Compare metrics across model types

## Model Parameters

### Classification Models
- **SVC:** kernel=["rbf", "linear"], C=[0.1, 1, 10], gamma=["scale", "auto"]
- **Random Forest:** n_estimators=[50, 100, 200], max_depth=[None, 5, 10]
- **Logistic Regression:** penalty=["l2"], solver=["saga", "lbfgs"], C=[0.1, 1, 10], max_iter=[500, 1000, 2000]

### Dimensionality Reduction
- **PCA:** n_components=[50, 100, 500]
- **t-SNE:** perplexity=[30, 50]
- **UMAP:** n_neighbors=[15, 30], min_dist=[0.1, 0.5]

## Results Summary

The combination of SVC with RBF kernel and PCA dimensionality reduction (500 components) achieved the best performance with 64% cross-validation accuracy. Dimensionality reduction improved efficiency without sacrificing performance, with PCA+SVC offering the best balance for CIFAR-10 classification. Non-linear models consistently outperformed linear approaches, highlighting the inherent complexity of image classification tasks.


