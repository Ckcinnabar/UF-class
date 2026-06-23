[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/dXJkvGaw)

# Speech Emotion Recognition System

A deep learning system that classifies speech audio clips according to emotional tone using mel-spectrogram analysis and hybrid CNN-LSTM architecture. The system achieves 87.41% test accuracy on a balanced 5-class emotion dataset.

## Project Overview

This project implements a speech emotion recognition system using two complementary approaches:

1. **CNN-LSTM Deep Learning Model** (Primary) - 87.41% test accuracy
2. **Random Forest Classifier** (Baseline) - 77.78% test accuracy

The system processes 3-second audio clips (48kHz sampling rate) and classifies them into 5 emotion categories using mel-spectrogram features extracted via the librosa library.

### Applications
- Human-computer interaction
- Sentiment analysis in call centers
- Mental health monitoring
- Accessibility tools for emotion recognition

## Team Members

- Haichen Fan - [@Haichen-Git](https://github.com/Haichen-Git)
- Zixuan Liu - [@zixuan-zoey](https://github.com/zixuan-zoey)
- Kuan-Chen Chen - [@Ckcinnabar](https://github.com/Ckcinnabar)

## Repository Structure

```
.
├── Data/
│   ├── ML_AllData.zip                    # Original audio dataset
│   ├── training_data_projectC.npy        # Preprocessed audio data (900×144000)
│   ├── training_labels_projectC.npy      # Emotion labels (900 samples)
│   └── Project C Speech Emotion Recognition - Training Dataset.ipynb
├── notebook/
│   ├── PyTorch_CNN_LSTM_EmotionRecognition.ipynb  # Main CNN-LSTM implementation
│   └── RandomForest_HaichenFan.ipynb              # Baseline Random Forest model
├── Models/                               # Saved model checkpoints
├── best_cnn_lstm_model_v3.pth           # Best trained model (87.41% accuracy)
└── README.md
```

## Dataset

- **Total Samples**: 900 audio clips
- **Audio Format**: 3-second mono audio at 48kHz (144,000 samples per clip)
- **Classes**: 5 emotion categories (balanced distribution ~20% each)
- **Split**: 70% training, 15% validation, 15% test
- **Augmentation**: 5x augmentation applied (noise injection, time shifting, volume scaling)

## Model Architecture

### CNN-LSTM Hybrid Model

The primary model combines convolutional and recurrent layers:

**Feature Extraction:**
- Mel-spectrogram (128 mel bands, 282 time frames)
- Audio processing: FFT window=1024, hop length=512, max frequency=8kHz

**Architecture:**
- 3 CNN layers (32→64→128 filters) with batch normalization and max pooling
- 2-layer bidirectional LSTM (128 hidden units per direction)
- Fully connected layers (256→128→5)
- Dropout (0.3) for regularization

**Training:**
- Optimizer: Adam (lr=0.001, weight decay=1e-5)
- Loss: Cross-entropy
- Scheduler: ReduceLROnPlateau
- Early stopping: patience=35 epochs
- Hardware: NVIDIA GeForce RTX 3060 Laptop GPU

### Random Forest Baseline

- Enhanced feature set: 1024 features (mean, std, max, min, median, quartiles, skewness)
- Hyperparameters: 150 trees, max_depth=25, entropy criterion
- Feature standardization using StandardScaler

## Results

| Model | Test Accuracy | Training Accuracy |
|-------|--------------|-------------------|
| CNN-LSTM | **87.41%** | ~99% |
| Random Forest (tuned) | 77.78% | 100% |
| Random Forest (baseline) | 71.11% | 100% |

### CNN-LSTM Per-Class Performance

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Emotion 1 | 0.862 | 0.893 | 0.877 | 28 |
| Emotion 2 | 0.920 | 0.852 | 0.885 | 27 |
| Emotion 3 | 0.952 | 0.741 | 0.833 | 27 |
| Emotion 4 | 0.833 | 0.962 | 0.893 | 26 |
| Emotion 5 | 0.833 | 0.926 | 0.877 | 27 |
| **Macro Avg** | **0.880** | **0.875** | **0.873** | **135** |

## Installation & Usage

### Requirements

```bash
pip install torch numpy librosa matplotlib scikit-learn jupyter tqdm seaborn
```

### Running the Model

1. **Extract features and train CNN-LSTM:**
   ```bash
   jupyter notebook notebook/PyTorch_CNN_LSTM_EmotionRecognition.ipynb
   ```

2. **Run Random Forest baseline:**
   ```bash
   jupyter notebook notebook/RandomForest_HaichenFan.ipynb
   ```

3. **Load pretrained model:**
   ```python
   import torch
   from model import CNN_LSTM_EmotionClassifier

   model = CNN_LSTM_EmotionClassifier(num_classes=5)
   model.load_state_dict(torch.load('best_cnn_lstm_model_v3.pth'))
   model.eval()
   ```

## Key Techniques

- **Mel-spectrogram representation** for audio feature extraction
- **Data augmentation** (5x): Gaussian noise, time shifting, volume scaling
- **Hybrid CNN-LSTM architecture** combining spatial and temporal modeling
- **Stratified train/val/test split** maintaining class balance
- **Early stopping and LR scheduling** to prevent overfitting
- **Batch normalization and dropout** for regularization

## References

1. **Librosa: Audio and Music Signal Analysis in Python**
   McFee, B., et al. (2015). librosa: Audio and Music Signal Analysis in Python. *Proceedings of the 14th Python in Science Conference*, 18-25.
   https://librosa.org/

2. **PyTorch: An Imperative Style, High-Performance Deep Learning Library**
   Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems*, 32.
   https://pytorch.org/

3. **Speech Emotion Recognition Using Deep Learning**
   Kwon, S. (2020). A CNN-Assisted Enhanced Audio Signal Processing for Speech Emotion Recognition. *Sensors*, 20(1), 183.
   https://doi.org/10.3390/s20010183

4. **LSTM Networks for Speech Emotion Recognition**
   Zhao, J., Mao, X., & Chen, L. (2019). Speech emotion recognition using deep 1D & 2D CNN LSTM networks. *Biomedical Signal Processing and Control*, 47, 312-323.
   https://doi.org/10.1016/j.bspc.2018.08.035

5. **Mel-Frequency Cepstral Coefficients for Audio Processing**
   Logan, B. (2000). Mel Frequency Cepstral Coefficients for Music Modeling. *International Symposium on Music Information Retrieval*.

6. **Random Forest Classifier**
   Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
   https://doi.org/10.1023/A:1010933404324

7. **Scikit-learn: Machine Learning in Python**
   Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
   https://scikit-learn.org/

