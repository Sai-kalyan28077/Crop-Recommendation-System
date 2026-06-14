# 🌾 Crop Recommendation System — Full Documentation

> **Internship Project — AI/ML based Crop Recommendation System**
> **Technologies:** Python, Streamlit, Pandas, Matplotlib, Seaborn, Scikit-learn

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [Project Structure](#3-project-structure)
4. [Installation & Setup](#4-installation--setup)
5. [File 1: `requirements.txt` — Line-by-Line Explanation](#5-file-1-requirementstxt--line-by-line-explanation)
6. [File 2: `model_training.py` — Line-by-Line Explanation](#6-file-2-model_trainingpy--line-by-line-explanation)
7. [File 3: `app.py` — Line-by-Line Explanation](#7-file-3-apppy--line-by-line-explanation)
8. [Machine Learning Concepts Explained](#8-machine-learning-concepts-explained)
9. [How to Run the Project](#9-how-to-run-the-project)
10. [Results & Output](#10-results--output)
11. [Future Enhancements](#11-future-enhancements)

---

## 1. Project Overview

### 1.1 What is this project?

The **Crop Recommendation System** is an AI-powered application that predicts the most suitable crop to grow based on soil nutrient levels (Nitrogen, Phosphorus, Potassium), environmental conditions (Temperature, Humidity, pH, Rainfall). The system uses machine learning classification algorithms trained on agricultural data to give farmers and agriculturalists data-driven crop suggestions.

### 1.2 Why is this needed?

- **Improve crop yield** — Choosing the right crop for given soil conditions maximizes productivity
- **Reduce guesswork** — Farmers often rely on intuition; ML provides scientific recommendations
- **Sustainable farming** — Optimizes resource utilization (water, fertilizers)
- **Quick decision support** — Get instant recommendations via an interactive web app

### 1.3 Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.10+** | Core programming language |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computing |
| **Matplotlib & Seaborn** | Data visualization (EDA) |
| **Scikit-learn** | Machine learning models & preprocessing |
| **Joblib** | Model serialization (save/load) |
| **Streamlit** | Web application framework |

---

## 2. Dataset Description

### 2.1 Source
The dataset used is `Crop_recommendation.csv` containing **2200 records** with **7 numerical features** and **1 categorical target** (22 crop classes).

### 2.2 Features (Independent Variables)

| Feature | Unit | Description | Range in Dataset |
|---------|------|-------------|------------------|
| **N** | kg/ha | Nitrogen content in soil | 0 – 140 |
| **P** | kg/ha | Phosphorus content in soil | 0 – 145 |
| **K** | kg/ha | Potassium content in soil | 0 – 205 |
| **temperature** | °C | Average temperature | 8.8 – 44.0 |
| **humidity** | % | Relative humidity | 14.3 – 99.98 |
| **ph** | — | Soil pH value | 3.5 – 10.0 |
| **rainfall** | mm | Annual rainfall | 20.2 – 298.6 |

### 2.3 Target (Dependent Variable)

**label** — The crop name. 22 crop classes:

`rice`, `maize`, `chickpea`, `kidneybeans`, `pigeonpeas`, `mothbeans`, `mungbean`, `blackgram`, `lentil`, `pomegranate`, `banana`, `mango`, `grapes`, `watermelon`, `muskmelon`, `apple`, `orange`, `papaya`, `coconut`, `cotton`, `jute`, `coffee`

Each crop has approximately **100 samples** (balanced dataset).

---

## 3. Project Structure

```
crop_recommendation_system/
│
├── requirements.txt                           # Python dependencies
├── model_training.py                          # ML model training script
├── app.py                                     # Streamlit web application
├── Crop_Recommendation_System_Documentation.md  # This document
├── Crop_recommendation.csv                    # Raw dataset
│
├── crop_recommendation_model.pkl              # Trained model (generated)
├── scaler.pkl                                 # StandardScaler (generated)
├── label_encoder.pkl                          # LabelEncoder (generated)
├── feature_distributions.png                  # EDA chart (generated)
├── eda_crop_analysis.png                      # EDA chart (generated)
├── correlation_matrix.png                     # EDA chart (generated)
├── pairplot.png                               # EDA chart (generated)
├── model_comparison.png                       # Model comparison chart (generated)
```

---

## 4. Installation & Setup

### 4.1 Prerequisites
- Python 3.10 or higher installed
- pip (Python package manager)

### 4.2 Step-by-Step Setup

```bash
# 1. Navigate to the project directory
cd crop_recommendation_system

# 2. (Recommended) Create a virtual environment
python -m venv venv
.\venv\Scripts\activate      # On Windows

# 3. Install required packages
pip install -r requirements.txt

# 4. Train the model (generates .pkl files and .png charts)
python model_training.py

# 5. Run the Streamlit app
streamlit run app.py
```

---

## 5. File 1: `requirements.txt` — Line-by-Line Explanation

```
streamlit==1.28.0
```
**Purpose:** Installs Streamlit, the web framework used to build the interactive UI. Version 1.28.0 ensures compatibility. Streamlit allows us to create a data app with Python alone — no HTML, CSS, or JavaScript needed.

```
pandas==2.0.3
```
**Purpose:** Pandas is the data manipulation library. We use it to load the CSV dataset into a DataFrame, explore data, filter, group, and prepare data for ML. Version 2.0.3 offers improved performance and stability.

```
numpy==1.24.3
```
**Purpose:** NumPy provides fast numerical array operations. It's the foundation for most scientific computing in Python. Scikit-learn internally uses NumPy arrays. We use it for statistical calculations and array manipulations.

```
matplotlib==3.7.2
```
**Purpose:** Matplotlib is the core plotting library in Python. We use it to create histograms, bar charts, box plots, and model comparison visualizations for Exploratory Data Analysis (EDA).

```
seaborn==0.12.2
```
**Purpose:** Seaborn is built on top of Matplotlib and provides high-level statistical visualizations. We use it for the correlation heatmap and pairplot — two essential EDA visualizations.

```
scikit-learn==1.3.0
```
**Purpose:** Scikit-learn is the machine learning library. It provides:
- `train_test_split` — Splits data into training and testing sets
- `StandardScaler` — Standardizes features (mean=0, std=1)
- `LabelEncoder` — Converts crop names into numeric labels
- 7 different classifier algorithms
- `accuracy_score`, `classification_report`, `confusion_matrix` — Evaluation metrics
- `cross_val_score` — Cross-validation for robust evaluation

```
joblib==1.3.2
```
**Purpose:** Joblib is used for efficient serialization of Python objects. We save (`dump`) and load (`load`) the trained model, scaler, and label encoder as `.pkl` files so they can be reused without retraining.

---

## 6. File 2: `model_training.py` — Line-by-Line Explanation

### Section 6.1: Import Statements

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
```
- `pandas as pd` — Imports Pandas with the alias `pd`. Used for DataFrame operations.
- `numpy as np` — Imports NumPy with alias `np`. Used for numerical arrays and mathematical operations.
- `matplotlib.pyplot as plt` — Imports pyplot for creating static, animated, and interactive visualizations.
- `seaborn as sns` — Imports Seaborn for statistical data visualization (heatmaps, pairplots).
- `joblib` — Imports Joblib for saving/loading Python objects (models, transformers).
- `os` — Provides operating system functions like creating directories and joining file paths.

```python
from sklearn.model_selection import train_test_split, cross_val_score
```
- `train_test_split` — Splits arrays or matrices into random train and test subsets. We use 80% for training, 20% for testing.
- `cross_val_score` — Evaluates model performance using k-fold cross-validation (more reliable than a single train-test split).

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler
```
- `LabelEncoder` — Converts categorical labels (crop names like 'rice', 'maize') into numeric values (0, 1, 2, ...). ML models need numerical inputs.
- `StandardScaler` — Standardizes features by removing the mean and scaling to unit variance. This ensures features with larger ranges (like rainfall 0-300) don't dominate features with smaller ranges (like pH 3.5-10).

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
```
These import 7 different classification algorithms. Each has a unique mathematical approach:
- **LogisticRegression** — Despite the name, it's used for classification. Uses the logistic function to model class probabilities.
- **DecisionTreeClassifier** — Creates a tree of if-else rules based on feature thresholds.
- **RandomForestClassifier** — Ensemble of many decision trees; averages their predictions.
- **GradientBoostingClassifier** — Builds trees sequentially, each correcting the previous ones' errors.
- **SVC (Support Vector Classifier)** — Finds the optimal hyperplane that separates classes.
- **KNeighborsClassifier** — Classifies based on the majority class among k nearest neighbors.
- **GaussianNB** — Applies Bayes' theorem assuming features follow a Gaussian distribution.

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```
- `accuracy_score` — Calculates the ratio of correct predictions to total predictions.
- `classification_report` — Generates precision, recall, f1-score, and support for each class.
- `confusion_matrix` — Shows a matrix of actual vs predicted class counts.

### Section 6.2: Data Loading & Initial Exploration

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Crop_recommendation.csv")
MODEL_DIR = BASE_DIR
```
- `BASE_DIR` — Resolves to the directory containing `model_training.py`, ensuring portability.
- `DATA_PATH` — Relative path to the CSV dataset constructed using `os.path.join`.
- `MODEL_DIR` — Same as `BASE_DIR`; generated files are saved alongside the script.

```python
df = pd.read_csv(DATA_PATH)
```
- `pd.read_csv()` — Reads the CSV file into a Pandas DataFrame (a 2D tabular data structure). The first row becomes column headers automatically.

```python
print(f"Dataset shape: {df.shape}")
```
- `df.shape` — Returns a tuple `(rows, columns)`. For our dataset: (2200, 8). 2200 samples and 8 columns (7 features + 1 label).

```python
print(f"Columns: {list(df.columns)}")
```
- `df.columns` — Returns column names as an Index object. Converted to list for display: `['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']`.

```python
print(f"Crops: {df['label'].nunique()}")
```
- `df['label'].nunique()` — Counts unique values in the 'label' column. Returns 22 (crop types).

```python
print(f"Crop types: {sorted(df['label'].unique())}")
```
- `df['label'].unique()` — Returns array of unique crop names. `sorted()` sorts them alphabetically.

```python
print(f"\nMissing values:\n{df.isnull().sum()}")
```
- `df.isnull()` — Returns a boolean DataFrame (True for missing values). `.sum()` counts Trues per column. This checks data quality — no missing values in our dataset.

```python
print(f"\nDataset info:")
print(df.info())
```
- `df.info()` — Prints a concise summary: column names, non-null counts, data types (dtypes), and memory usage.

```python
print(f"\nStatistical summary:\n{df.describe())
```
- `df.describe()` — Generates descriptive statistics (count, mean, std, min, 25%, 50%, 75%, max) for all numerical columns.

### Section 6.3: Exploratory Data Analysis (EDA) — Visualizations

#### Histograms (Feature Distributions)

```python
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
```
- `plt.subplots(3, 3)` — Creates a 3×3 grid of subplots (9 plots total). Returns figure and axes array.
- `.flatten()` — Flattens the 2D axes array into 1D for easy iteration.

```python
numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
```
- Iterates through each numeric column.
- `axes[i].hist()` — Creates a histogram with 30 bins, black edges, and 70% opacity.
- `set_title()`, `set_xlabel()`, `set_ylabel()` — Adds labels to the plot.
- Histograms help us see the data distribution shape (e.g., normal, skewed, bimodal).

```python
axes[-1].axis('off')
```
- Hides the 9th subplot since we only have 7 features.

```python
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'feature_distributions.png'), dpi=150)
plt.close()
```
- `tight_layout()` — Automatically adjusts subplot spacing to prevent overlap.
- `savefig()` — Saves the figure to a PNG file at 150 DPI resolution.
- `close()` — Closes the figure to free memory.

#### Box Plots (Crop-wise Feature Analysis)

```python
fig, axes = plt.subplots(4, 2, figsize=(14, 16))
axes = axes.flatten()
```
- Creates an 4×2 grid of subplots (8 total).

```python
crop_counts = df['label'].value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(crop_counts)))
axes[0].bar(crop_counts.index, crop_counts.values, color=colors)
axes[0].set_title('Crop Distribution in Dataset')
axes[0].tick_params(axis='x', rotation=45)
```
- `value_counts()` — Counts occurrences of each crop. Since the dataset is balanced, each crop has ~100 samples.
- `plt.cm.Set3` — Uses the Set3 colormap, generating distinct colors for each bar.
- `bar()` — Creates a bar chart showing crop distribution.
- `tick_params(rotation=45)` — Rotates x-axis labels for readability.

```python
for i, col in enumerate(numeric_cols):
    df.boxplot(column=col, by='label', ax=axes[i+1], rot=45, fontsize=8)
    axes[i+1].set_title(f'{col} by Crop')
```
- Creates box plots showing the distribution of each feature grouped by crop type.
- Box plots show: median (middle line), Q1/Q3 (box edges), min/max (whiskers), and outliers (dots).
- This helps identify which features differentiate crops. For example, rice prefers high rainfall while chickpea prefers low rainfall.

#### Correlation Heatmap

```python
corr = df[numeric_cols].corr()
```
- `.corr()` — Computes pairwise correlation coefficients (Pearson's r) between all numeric columns. Values range from -1 (strong negative correlation) to +1 (strong positive correlation).

```python
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Matrix')
```
- `sns.heatmap()` — Creates a color-coded heatmap of the correlation matrix.
- `annot=True` — Displays the correlation values in each cell.
- `cmap='coolwarm'` — Red for positive, blue for negative correlation.
- `fmt='.2f'` — Formats values to 2 decimal places.
- `linewidths=0.5` — Adds lines between cells.
- The heatmap helps identify multicollinearity (features that are highly correlated with each other).

#### Pairplot

```python
sns.pairplot(df[numeric_cols + ['label']], hue='label', palette='Set2')
```
- `pairplot()` — Creates a matrix of scatter plots for every pair of features, colored by crop type.
- `hue='label'` — Colors points by crop class.
- `palette='Set2'` — Uses a colorblind-friendly palette.
- Diagonal shows KDE (Kernel Density Estimation) plots.
- This is the most comprehensive EDA visualization — shows relationships between all feature pairs.

### Section 6.4: Data Preprocessing

```python
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])
```
- Creates a LabelEncoder instance.
- `fit_transform()` first `fit`s (learns the mapping from crop names to numbers) and then `transform`s (converts names to numbers).
- Example mapping: {'rice': 0, 'maize': 1, 'chickpea': 2, ...}.
- Now the target is numeric and can be used by ML algorithms.

```python
joblib.dump(label_encoder, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
```
- Saves the encoder to a file. When the web app makes predictions, it loads this encoder to convert numeric predictions back to crop names.

```python
X = df[numeric_cols].values
y = df['label_encoded'].values
```
- `X` — Feature matrix (2200 × 7 NumPy array). All input variables.
- `y` — Target vector (2200-element array). The encoded crop labels.
- `.values` converts Pandas DataFrame/Series to NumPy arrays for scikit-learn compatibility.

### Section 6.5: Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
- Splits data into:
  - **X_train** (1760 samples, 80%) — Used to train the model
  - **X_test** (440 samples, 20%) — Held back for final evaluation
  - **y_train**, **y_test** — Corresponding labels
- `test_size=0.2` — 20% of data goes to test set.
- `random_state=42` — Seed for reproducibility. Same seed always gives the same split.
- `stratify=y` — Ensures the class distribution in train and test sets matches the original distribution. Critical for balanced evaluation.

### Section 6.6: Feature Scaling

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
- `StandardScaler()` — Computes mean (μ) and standard deviation (σ) for each feature.
- `fit_transform(X_train)` — Learns μ and σ from training data, then transforms training data: `z = (x - μ) / σ`.
- `transform(X_test)` — Applies the *same* transformation (using training μ and σ) to test data. Never `fit` on test data — that would cause data leakage.
- Why scale? Algorithms like SVM, KNN, and Logistic Regression use distance calculations. Features with large values (e.g., rainfall ~300) would dominate features with small values (e.g., pH ~7) without scaling. Tree-based models (Decision Tree, Random Forest) are not affected by scaling.

```python
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
```
- Saves the scaler for use in the web app. User inputs must be scaled the same way as training data.

### Section 6.7: Model Training & Evaluation

```python
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42, probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB()
}
```
- Creates a dictionary mapping model names to their instances.
- **LogisticRegression**: `max_iter=1000` allows enough iterations for convergence. Works well when classes are linearly separable.
- **DecisionTreeClassifier**: Splits data recursively based on feature thresholds (Gini impurity or entropy). Easy to interpret.
- **RandomForestClassifier**: `n_estimators=100` builds 100 decision trees and averages them. Reduces overfitting compared to a single tree.
- **GradientBoostingClassifier**: Builds trees sequentially (100 trees), each tree tries to correct the errors of the previous ensemble.
- **SVC**: `kernel='rbf'` (Radial Basis Function) maps data to higher dimensions for non-linear separation. `probability=True` enables probability estimates.
- **KNeighborsClassifier**: `n_neighbors=5` looks at 5 nearest data points and takes a majority vote.
- **GaussianNB**: Assumes features follow a normal (Gaussian) distribution and applies Bayes' theorem.

```python
results = []
best_model = None
best_accuracy = 0
```
- Initializes tracking variables for model comparison.

```python
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
```
- Iterates through each model.
- `.fit()` trains the model on the scaled training data. Each model learns its internal parameters (e.g., decision tree splits, SVM support vectors, logistic regression coefficients).

```python
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
```
- `.predict()` generates predictions on the unseen test data.
- `accuracy_score()` computes: (correct predictions / total predictions) × 100.

```python
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    cv_mean = cv_scores.mean()
```
- **Cross-validation**: Splits training data into 5 folds. Trains on 4 folds, validates on 1 fold. Repeats 5 times. This gives a more reliable accuracy estimate than a single train-test split.
- `cv_scores.mean()` — Average accuracy across all 5 folds.

```python
    print(f"Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=label_encoder.classes_))
```
- `classification_report()` generates per-class metrics:
  - **Precision** — Of all predicted "rice" samples, how many were actually rice? `TP / (TP + FP)`
  - **Recall** — Of all actual "rice" samples, how many did we correctly predict? `TP / (TP + FN)`
  - **F1-score** — Harmonic mean of precision and recall. `2 × (Precision × Recall) / (Precision + Recall)`
  - **Support** — Number of actual samples for each class in the test set.

```python
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name
```
- Tracks the best performing model based on test accuracy.

### Section 6.8: Model Comparison Visualization

```python
results_df = pd.DataFrame(results)
```
- Converts the results list into a DataFrame for easy display.

```python
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(results_df['Model'], results_df['Test Accuracy'].astype(float),
              color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#D4A5A5', '#9B59B6'])
```
- Creates a bar chart comparing model accuracies.
- Each bar gets a different color for visual distinction.

```python
ax.set_ylim(0, 1.0)
for bar, acc in zip(bars, results_df['Test Accuracy'].astype(float)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')
```
- `set_ylim(0, 1.0)` — Sets y-axis from 0 to 1 (0% to 100%).
- Annotates each bar with its accuracy percentage.

### Section 6.9: Save Best Model

```python
joblib.dump(best_model, os.path.join(MODEL_DIR, 'crop_recommendation_model.pkl'))
```
- Serializes and saves the best performing model to disk.
- The `.pkl` file contains all learned parameters and can be loaded later for predictions without retraining.

---

## 7. File 3: `app.py` — Line-by-Line Explanation

### Section 7.1: Imports

```python
import streamlit as st
```
- Imports Streamlit as `st`. Streamlit provides functions to create web UI elements:
  - `st.sidebar.slider()` — Sidebar slider widget
  - `st.button()` — Clickable button
  - `st.dataframe()` — Display Pandas DataFrame
  - `st.image()` — Display images
  - `st.pyplot()` — Display Matplotlib figures
  - `st.markdown()` — Render Markdown text
  - `st.info()`, `st.success()` — Info/success message boxes
  - `st.tabs()` — Tabbed containers

```python
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
```
- Standard libraries for data manipulation, model loading, and visualization (same as in model_training.py).

### Section 7.2: Load Saved Artifacts

```python
base_dir = os.path.dirname(__file__)
```
- `base_dir` — Resolves to the directory containing `app.py` so model files are loaded from the same location regardless of where the project is moved.

```python
model = joblib.load(os.path.join(base_dir, 'crop_recommendation_model.pkl'))
scaler = joblib.load(os.path.join(base_dir, 'scaler.pkl'))
label_encoder = joblib.load(os.path.join(base_dir, 'label_encoder.pkl'))
```
- `joblib.load()` — Deserializes the saved model, scaler, and label encoder.
- These are now ready for inference:
  - `scaler` — Transforms user inputs to the same scale as training data.
  - `model` — The trained classifier that predicts the crop.
  - `label_encoder` — Converts numeric predictions back to human-readable crop names.

### Section 7.3: Page Configuration

```python
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)
```
- `set_page_config()` — Sets browser tab title, favicon emoji, and layout.
- `layout="wide"` — Uses the full browser width for the app.

### Section 7.4: Custom CSS Styling

```python
st.markdown("""
<style>
    .main-header { text-align: center; padding: 1rem; }
    .main-header h1 { color: #2E7D32; font-size: 2.5rem; }
    .prediction-box { padding: 1.5rem; border-radius: 10px; text-align: center; }
    .crop-name { font-size: 2rem; font-weight: bold; color: #1B5E20; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 10px; text-align: center; }
    .metric-value { font-size: 1.5rem; font-weight: bold; color: #2E7D32; }
    .metric-label { font-size: 0.9rem; color: #666; }
</style>
""", unsafe_allow_html=True)
```
- `st.markdown()` with CSS inside `<style>` tags applies custom styling.
- `unsafe_allow_html=True` allows HTML/CSS in Markdown.
- Styles the header, prediction box, crop name, and metric cards with a green agricultural theme.

### Section 7.5: App Header

```python
st.markdown('<div class="main-header"><h1>🌾 Crop Recommendation System</h1></div>',
            unsafe_allow_html=True)
```
- Creates a centered header with an emoji icon and green color (from CSS).

```python
st.markdown("""
<p style="text-align:center; font-size:1.1rem; color:#555;">
AI-powered system that recommends the most suitable crop based on your soil and environmental conditions.
</p>
""", unsafe_allow_html=True)
```
- Subtitle description of the application.

### Section 7.6: Sidebar — User Inputs

```python
st.sidebar.header("📊 Input Parameters")
st.sidebar.markdown("Adjust the sliders to match your soil and weather conditions.")
```
- Sidebar header and instruction text.

```python
N = st.sidebar.slider("Nitrogen (N)", 0, 140, 50,
                       help="Nitrogen content in soil (kg/ha)")
```
- `st.sidebar.slider()` creates a slider widget in the sidebar.
- Parameters: (label, min_value, max_value, default_value).
- `help` — Tooltip text shown on hover.
- Each slider corresponds to one feature in the dataset:
  - N: 0 to 140, default 50
  - P: 0 to 145, default 50
  - K: 0 to 210, default 50
  - Temperature: 8.0 to 45.0, default 25.0
  - Humidity: 14.0 to 100.0, default 60.0
  - pH: 3.5 to 10.0, default 6.5
  - Rainfall: 20.0 to 300.0, default 150.0

```python
input_data = pd.DataFrame({
    'N': [N], 'P': [P], 'K': [K],
    'temperature': [temperature], 'humidity': [humidity],
    'ph': [ph], 'rainfall': [rainfall]
})
```
- Creates a single-row DataFrame from the slider values. This matches the format expected by our scaler and model.

### Section 7.7: Main Layout — Input Display & Instructions

```python
col1, col2 = st.columns([2, 1])
```
- Creates two columns with a 2:1 width ratio.

```python
with col1:
    st.subheader("📋 Input Soil & Weather Conditions")
    st.dataframe(input_data, use_container_width=True)
```
- Displays the user's input values in a formatted DataFrame table.

```python
with col2:
    st.subheader("ℹ️ Instructions")
    st.info("""
    1. Adjust the values on the left sidebar
    2. Click **Predict Crop** below
    3. Get an instant recommendation!
    """)
```
- Shows usage instructions in an info box.

### Section 7.8: Prediction Logic

```python
if st.button("🌱 Predict Recommended Crop", type="primary", use_container_width=True):
```
- `st.button()` creates a clickable button with `type="primary"` (highlighted style) and `use_container_width=True` (full width).

```python
    input_scaled = scaler.transform(input_data.values)
```
- `input_data.values` extracts the NumPy array from the DataFrame.
- `scaler.transform()` applies the same standardization learned during training: `z = (x - μ) / σ`.

```python
    prediction = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)[0]
```
- `model.predict()` returns the predicted class index (e.g., 14 for 'rice').
- `model.predict_proba()` returns probability scores for ALL 22 classes. `[0]` gets the first (and only) row since we're predicting for one sample.

```python
    crop_name = label_encoder.inverse_transform(prediction)[0]
```
- `label_encoder.inverse_transform()` converts numeric prediction back to the original crop name string.

```python
    confidence = np.max(probabilities) * 100
```
- `np.max()` finds the highest probability among all classes. Multiplying by 100 converts to percentage.

### Section 7.9: Display Prediction Result

```python
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        st.markdown(f"""
        <div class="prediction-box" style="background: linear-gradient(135deg, #E8F5E9, #C8E6C9);">
            <p style="font-size:1.2rem; color:#555;">Recommended Crop</p>
            <p class="crop-name">🌿 {crop_name.upper()}</p>
            <p style="font-size:1.1rem; color:#388E3C;">
                Confidence: {confidence:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
```
- Centers the prediction result in a green gradient box.
- Shows the recommended crop name in large, bold text.
- Shows the confidence score (how sure the model is about its prediction).

### Section 7.10: Probability Visualization

```python
    prob_df = pd.DataFrame({
        'Crop': label_encoder.classes_,
        'Probability (%)': probabilities * 100
    }).sort_values('Probability (%)', ascending=False).reset_index(drop=True)
```
- Creates a DataFrame pairing each crop name with its predicted probability.
- `sort_values()` — Orders from highest to lowest probability.
- `reset_index(drop=True)` — Resets the index after sorting.

```python
    with col_p1:
        fig, ax = plt.subplots(figsize=(10, 6))
        top_n = prob_df.head(10)
        colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(top_n)))
        bars = ax.barh(top_n['Crop'][::-1], top_n['Probability (%)'][::-1], color=colors[::-1])
```
- Creates a horizontal bar chart of the top 10 most likely crops.
- `[::-1]` reverses the order so the highest probability appears at the top.
- `plt.cm.Greens` uses a green gradient colormap, fitting the agricultural theme.

```python
        for bar, prob in zip(bars, top_n['Probability (%)'][::-1]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{prob:.1f}%', va='center', fontsize=10)
```
- Annotates each bar with its percentage value.

### Section 7.11: EDA Visualization Tabs

```python
tab1, tab2, tab3 = st.tabs(["Feature Distributions", "Crop Analysis", "Feature Insights"])
```
- Creates three tabs for different EDA visualizations.

**Tab 1 — Feature Distributions:**
```python
with tab1:
    st.image(
        os.path.join(MODEL_DIR, 'feature_distributions.png'),
        use_column_width=True,
        caption="Distribution of each feature in the dataset"
    )
```
- Displays the histogram chart generated during training.
- `use_column_width=True` — Scales image to fit the column width.

**Tab 2 — Crop Analysis:**
```python
with tab2:
    st.image(
        os.path.join(MODEL_DIR, 'eda_crop_analysis.png'),
        use_column_width=True,
        caption="Crop-wise analysis of each feature"
    )
```
- Displays the box plot chart showing feature distributions per crop.

**Tab 3 — Feature Insights:**
```python
with tab3:
    st.image(
        os.path.join(MODEL_DIR, 'model_comparison.png'),
        use_column_width=True,
        caption="Accuracy comparison of different ML models"
    )
```
- Shows the model accuracy comparison bar chart.

### Section 7.12: Footer

```python
st.markdown("""
<div style="text-align:center; color:#888; padding:1rem;">
    <p>📌 <strong>Crop Recommendation System</strong> | Powered by Machine Learning &amp; Streamlit</p>
    <p style="font-size:0.8rem;">Built with ❤️ for agricultural decision support</p>
</div>
""", unsafe_allow_html=True)
```
- Centered footer with attribution text.

---

## 8. Machine Learning Concepts Explained

### 8.1 Supervised Learning — Classification

This is a **multi-class classification** problem (22 classes). In supervised learning:
- We have labeled data (input features + correct crop name)
- The model learns the mapping: `features → crop`
- During prediction, it applies this mapping to new, unseen data

### 8.2 Train-Test Split

| Set | Size | Purpose |
|-----|------|---------|
| Training | 80% (1760 samples) | Model learns patterns from this data |
| Testing | 20% (440 samples) | Unseen data for final evaluation |

`stratify=y` ensures all 22 crops are proportionally represented in both sets.

### 8.3 Feature Scaling (Standardization)

`z = (x - μ) / σ`

- **μ** = mean of training feature
- **σ** = standard deviation of training feature
- **z** = standardized value (mean=0, std=1)

Without scaling, a feature like rainfall (0–300) would dominate a feature like pH (3.5–10) in distance-based algorithms.

### 8.4 Cross-Validation (k-fold, k=5)

Instead of one fixed validation set, data is split into 5 folds. The model is trained 5 times, each time using 4 folds for training and 1 fold for validation. Results are averaged. This reduces the variance of the accuracy estimate.

### 8.5 How Each Algorithm Works

#### Logistic Regression
- Despite the name, it's a classification algorithm
- Uses the sigmoid function: `P(y=1) = 1 / (1 + e^-(β₀ + β₁x₁ + ... + βₙxₙ))`
- For multi-class, uses "one-vs-rest" strategy

#### Decision Tree
- Recursively splits data at decision nodes based on feature thresholds
- Splits are chosen to maximize information gain (reduce impurity)
- Uses Gini impurity or entropy as splitting criteria
- Prone to overfitting if tree is too deep

#### Random Forest
- Builds 100 decision trees on random subsets of data (bootstrap samples)
- Each tree considers random subsets of features at each split
- Final prediction = majority vote of all trees
- Reduces overfitting and improves generalization

#### Gradient Boosting
- Builds trees **sequentially** (not independently like Random Forest)
- Each new tree tries to correct the errors (residuals) of the previous ensemble
- Uses gradient descent to minimize the loss function
- Often achieves higher accuracy than Random Forest

#### SVM (Support Vector Machine)
- Finds the hyperplane that best separates classes with maximum margin
- RBF kernel maps data to higher dimensions for non-linear separation
- Support vectors = data points closest to the decision boundary

#### KNN (K-Nearest Neighbors)
- Lazy learning: no explicit training phase
- For a new point, finds k=5 closest training points (by Euclidean distance)
- Prediction = majority class among those 5 neighbors
- Sensitive to feature scaling (hence we scaled)

#### Naive Bayes
- Based on Bayes' theorem: `P(A|B) = P(B|A) × P(A) / P(B)`
- "Naive" because assumes all features are independent (which is rarely true)
- GaussianNB assumes continuous features follow normal distribution
- Fast and works well with high-dimensional data

### 8.6 Evaluation Metrics

| Metric | Formula | What it measures |
|--------|---------|------------------|
| **Accuracy** | `(TP + TN) / (TP + TN + FP + FN)` | Overall correctness |
| **Precision** | `TP / (TP + FP)` | How many predicted positives are actually positive |
| **Recall** | `TP / (TP + FN)` | How many actual positives did we catch |
| **F1-Score** | `2 × (P × R) / (P + R)` | Harmonic mean of precision and recall |
| **Support** | — | Number of actual samples for each class |

Where TP=True Positive, TN=True Negative, FP=False Positive, FN=False Negative.

---

## 9. How to Run the Project

### Step 1: Install Dependencies
```bash
cd crop_recommendation_system
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
python model_training.py
```
This will:
- Load and analyze the dataset
- Generate EDA visualizations (saved as .png files)
- Train 7 machine learning models
- Compare their performance
- Save the best model, scaler, and label encoder as .pkl files

### Step 3: Launch the Web App
```bash
streamlit run app.py
```
This will:
- Start a local web server (typically at http://localhost:8501)
- Open the Crop Recommendation System in your default browser
- Allow you to adjust soil parameters and get instant predictions

---

## 10. Results & Output

### 10.1 Model Performance

After testing 7 classification algorithms, the model comparison chart shows accuracy scores. Typically, ensemble methods (Random Forest, Gradient Boosting) achieve the highest accuracy (95%+), while simpler models like Naive Bayes or Logistic Regression may score lower.

### 10.2 Sample Prediction

| Input Values | Predicted Crop | Confidence |
|-------------|----------------|------------|
| N=90, P=42, K=43, Temp=20.9°C, Hum=82%, pH=6.5, Rain=203mm | **Rice** | ~99% |
| N=71, P=54, K=16, Temp=22.6°C, Hum=63.7%, pH=5.75, Rain=87.8mm | **Maize** | ~99% |

### 10.3 Generated Files

| File | Description |
|------|-------------|
| `crop_recommendation_model.pkl` | Trained best ML model |
| `scaler.pkl` | Fitted StandardScaler |
| `label_encoder.pkl` | Fitted LabelEncoder |
| `feature_distributions.png` | Histograms of all features |
| `eda_crop_analysis.png` | Box plots per crop |
| `correlation_matrix.png` | Feature correlation heatmap |
| `pairplot.png` | Pairwise feature scatter plots |
| `model_comparison.png` | Bar chart comparing model accuracies |

---

## 11. Future Enhancements

1. **Deep Learning Integration** — Use Neural Networks (TensorFlow/Keras) for potentially higher accuracy
2. **More Features** — Add soil type (clay/sand/loam), season, region, past crop history
3. **Real-time Weather API** — Fetch live temperature/humidity/rainfall data
4. **Multi-language Support** — Localize for regional farmers
5. **Fertilizer Recommendation** — Suggest NPK fertilizer quantities based on soil deficit
6. **Crop Rotation Planning** — Suggest rotation schedules for sustainable farming
7. **Mobile App** — Deploy via Streamlit sharing or convert to mobile app
8. **Explainability** — Add SHAP/LIME to explain why a particular crop was recommended
9. **User Feedback Loop** — Allow farmers to provide feedback to improve the model
10. **Deployment** — Host on cloud platforms (Heroku, AWS, Azure, Streamlit Cloud)

---

> **End of Documentation**
>
> This project demonstrates the complete lifecycle of an ML project:
> 1. **Data Collection** — Agricultural dataset
> 2. **Exploratory Data Analysis** — Understanding patterns through visualizations
> 3. **Preprocessing** — Encoding, scaling, train-test splitting
> 4. **Model Training** — 7 different algorithms with cross-validation
> 5. **Evaluation** — Accuracy, precision, recall, F1-score
> 6. **Deployment** — Interactive Streamlit web application
> 7. **Documentation** — Complete line-by-line explanation
