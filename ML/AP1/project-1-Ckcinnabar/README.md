# Spotify Song Attributes Analysis

## Project Overview
This project analyzes music-related attributes from Spotify songs to build machine learning models for classification and regression tasks. The analysis explores how different song features relate to energy levels and danceability, providing insights into music characteristics and their relationships.

## Tasks
### Task 1. Provide a summary of findings from EDA (bullet points or short analysis)
Genre has the most null in the dataset,loudness have most outliers, and the tempo have the biggest range, but tempo (I guess) might not be a important varible, because compairing with the others,  correlation of tempo is near 0 to all other varible, so I'll keep it raw without any manipulate. 

### Task 2. Provide at least three visualizations showing trends or insights from the dataset. 
I provide correlation heatmap, scatter plot to observe Danceability and Loudness and I also use boxplot to find the outliers.

### Task 3. Provide a written summary of the preprocessing steps.

1. **Handling Missing Values**: During initial data exploration, we identified that the 'genre' column contained the most missing values. After filtering the dataset to include only records with non-null genre values (`df[df['genre'].notna()]`), we discovered that this operation effectively removed all missing values across the entire dataset, resulting in a complete dataset with no null values in any column.

2. **Outlier Removal**: To improve model performance and reliability, we implemented statistical outlier removal using the z-score method. Data points falling outside the 95% confidence interval (corresponding to a z-score threshold of 1.96) were excluded from the analysis. This approach helped prevent extreme values from negatively impacting our models.

3. **Feature Scaling and Encoding**:
   - Numerical features were normalized using `MinMaxScaler()` to ensure all values fall within a similar range, improving model convergence and performance
   - Categorical features (particularly 'genre') were transformed using `OneHotEncoder` to convert them into a format suitable for machine learning algorithms
   
4. **Train-Test Split**: The dataset was divided into training (80%) and testing (20%) sets using a random state of 2025 to ensure reproducibility of results.

### Task 4. Clearly state the target variable for both classification and regression AND Explain why this task is interesting.

 - Regression : I'm going to use `loudness`, `acousticness` and `tempo` to predict `energy`, by the correlation heatmap, we can know that both `loudness` and `acousticness` have very high value with `energy`, and I asume `tempo` is very important to `energy`.

 - Classification: use `energy`, `tempo`	and `loudness` to predict `danceability`

###  Task 5. After completing all steps above, provide the following:
1. A short explanation of which model performed better and why.
   - For the regression one, Decision Tree works better, because our feature set appears to have such non-linear patterns, making the Decision Tree naturally more suitable for this prediction task.
   - For the classification one, Random Forest Classifier performs well on our selected features as it combines multiple decision trees into an ensemble model, allowing it to capture complex patterns in the energy and loudness relationships while reducing overfitting through majority voting across trees.

2. Are there any differences when adding regularization into regression? Which features are more important?
- For my model, it didn't work so well, but for other topic, I think it might work better affter regularization.

### Task 6. Compare models and justify which one is better for each task.
Decision Tree works better for regression, Random Forest Classifier works better for classification.

### Task 7: Interpretation of Results 
- Observed Trends

   - Non-linear relationships dominate music features, with Decision Tree (R² = 0.9556) far outperforming linear models (R² = 0.6336)
Loudness emerged as the strongest predictor for both energy and danceability classification
Songs with high energy and loudness typically fall in medium-to-high danceability categories

- Model Generalization

   - Models maintained strong performance metrics on test data due to effective cross-validation
Classification models achieved good balance between accuracy and precision-recall trade-offs

- Applications for Music Streaming Platforms

   - Enable automatic generation of mood-based playlists based on audio characteristics
   - Improve recommendation systems by matching user preferences with similar audio features
   - Help content discovery by allowing users to find music with specific energy/danceability levels
- Provide business intelligence on emerging audio trends and regional preferences

   - The strong performance of non-linear models suggests streaming platforms should implement more sophisticated algorithms capable of capturing complex interactions between audio features.
 
   



## Dataset Description
The dataset (`Spotify_Song_Attributes.csv`) contains attributes of songs played on Spotify until 2022, including:

- **Track information**: trackName, artistName, duration, genre, etc.
- **Audio features**: danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature

Key attributes used in this analysis:
- `danceability`: Measure of how suitable a track is for dancing (0.0 to 1.0)
- `energy`: Measure of intensity and activity (0.0 to 1.0)
- `loudness`: Overall loudness in decibels (dB)
- `acousticness`: Measure of acoustic quality (0.0 to 1.0)
- `tempo`: Speed of the track in beats per minute (BPM)

## Installation Requirements

To run the code in this repository, you need the following dependencies:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

Install the required packages using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## File Structure

- `training.ipynb`: Jupyter notebook for data exploration, preprocessing, and model training
- `test.ipynb`: Jupyter notebook to evaluate trained models on test data
- `Spotify_Song_Attributes.csv`: Dataset file (must be in the same directory)
- `model_*.joblib`: Saved trained models
- `df_model.csv`: Preprocessed dataset used for model evaluation

### Trained Models
- `model_linear_reg.joblib`: Linear Regression model
- `model_Ridge.joblib`: Ridge Regression model
- `model_Lasso.joblib`: Lasso Regression model
- `model_DecisionTreeRegressor.joblib`: Decision Tree Regression model
- `model_LogisticRegression.joblib`: Logistic Regression classification model
- `RandomForestClassifier.joblib`: Random Forest classification model

## Usage Instructions

### Running the Training Notebook
1. Ensure `Spotify_Song_Attributes.csv` is in the same directory
2. Open and run `training.ipynb` to:
   - Perform exploratory data analysis
   - Clean and preprocess the data
   - Train regression and classification models
   - Save trained models to disk

### Running the Testing Notebook
1. Ensure all model files (*.joblib) are in the same directory
2. Open and run `test.ipynb` to:
   - Load the trained models
   - Evaluate performance on test data
   - Generate visualizations and metrics

## Machine Learning Tasks

### Regression Task
Using `loudness`, `acousticness`, and `tempo` to predict song `energy`

- **Models Implemented**:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Decision Tree Regressor

- **Evaluation Metrics**:
  - R² (Coefficient of Determination)
  - MAE (Mean Absolute Error)
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)

### Classification Task
Using `loudness`, `energy`, and `tempo` to categorize `danceability` into:
- Low (≤ 0.3)
- Medium (> 0.3 and ≤ 0.6)
- High (> 0.6)

- **Models Implemented**:
  - Logistic Regression
  - Random Forest Classifier

- **Evaluation Metrics**:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion Matrix
  - ROC-AUC Curve

## User-Defined Parameters

### Data Preprocessing Parameters
- **Z-score threshold**: Set to 1.96 for outlier removal (can be modified in the `remove_outliers_all_columns` function)
- **Train-test split**: 80% training, 20% testing (random_state=2025)
- **Danceability categorization**: 
  - Low: ≤ 0.3
  - Medium: > 0.3 and ≤ 0.6
  - High: > 0.6

### Model Hyperparameters
- **Ridge/Lasso regularization**: alpha values [0, 0.001, 0.01, 0.1, 0.2, 0.5, 1]
- **Decision Tree**: 
  - max_depth: [3, 6, 9, None]
  - min_samples_split: [2, 5, 10]
- **Logistic Regression**:
  - penalty: ['l2', None]
  - C: [0.1, 1.0, 10.0]
- **Random Forest**:
  - n_estimators: [50, 100, 200]
  - max_depth: [3, 5, 7, None]

## Results Summary

### Regression Performance
The Decision Tree Regressor demonstrates superior performance in predicting song energy levels, with significantly higher R² and lower error rates compared to linear models.

### Classification Performance
Both Logistic Regression and Random Forest Classifier show good performance in categorizing songs by danceability level, with Random Forest generally achieving higher precision and recall for all categories.

