import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "Crop_recommendation.csv")
MODEL_DIR = BASE_DIR

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Cols:", list(df.columns))
print("Unique crops:", df['label'].nunique())
print("Crop list:", sorted(df['label'].unique()))
print("\nMissing:")
print(df.isnull().sum())
print("\nStats:")
print(df.describe())

os.makedirs(MODEL_DIR, exist_ok=True)

cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, c in enumerate(cols):
    axes[i].hist(df[c], bins=30, edgecolor='black', alpha=0.7)
    axes[i].set_title(f'Distribution of {c}')
    axes[i].set_xlabel(c)
    axes[i].set_ylabel('Frequency')
axes[-1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'feature_distributions.png'), dpi=150)
plt.close()

fig, axes = plt.subplots(4, 2, figsize=(14, 16))
axes = axes.flatten()
cnt = df['label'].value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(cnt)))
axes[0].bar(cnt.index, cnt.values, color=colors)
axes[0].set_title('Crop Distribution in Dataset')
axes[0].set_xlabel('Crop')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

for i, c in enumerate(cols):
    df.boxplot(column=c, by='label', ax=axes[i+1], rot=45, fontsize=8)
    axes[i+1].set_title(f'{c} by Crop')
    axes[i+1].set_xlabel('')
axes[-1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'eda_crop_analysis.png'), dpi=150)
plt.close()

corr = df[cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'correlation_matrix.png'), dpi=150)
plt.close()

sns.pairplot(df[cols + ['label']], hue='label', palette='Set2')
plt.savefig(os.path.join(MODEL_DIR, 'pairplot.png'), dpi=150)
plt.close()

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])
joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))

X = df[cols].values
y = df['label_encoded'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

all_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42, probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB()
}

results = []
best = None
best_acc = 0

for name, clf in all_models.items():
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    cv = cross_val_score(clf, X_train_scaled, y_train, cv=5)
    cv_mean = cv.mean()

    results.append({
        'Model': name,
        'Test Accuracy': f"{acc:.4f}",
        'CV Accuracy': f"{cv_mean:.4f}"
    })

    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"CV Accuracy: {cv_mean:.4f} (+/- {cv.std():.4f})")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    if acc > best_acc:
        best_acc = acc
        best = clf
        best_name = name

results_df = pd.DataFrame(results)
print(f"\n\n{'='*50}")
print("Model Comparison Summary:")
print(results_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(results_df['Model'], results_df['Test Accuracy'].astype(float),
              color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#D4A5A5', '#9B59B6'])
ax.set_title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.tick_params(axis='x', rotation=45)
ax.set_ylim(0, 1.0)
for bar, acc in zip(bars, results_df['Test Accuracy'].astype(float)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'model_comparison.png'), dpi=150)
plt.close()

print(f"\n{'='*50}")
print(f"Best Model: {best_name}")
print(f"Best Test Accuracy: {best_acc:.4f}")

joblib.dump(best, os.path.join(MODEL_DIR, 'crop_recommendation_model.pkl'))
print(f"Saved: {os.path.join(MODEL_DIR, 'crop_recommendation_model.pkl')}")
print(f"All files in: {MODEL_DIR}")
