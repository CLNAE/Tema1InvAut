import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import drive
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

# === Montare Google Drive și încărcare dataset ===
drive.mount('/content/drive')
stroke_set = pd.read_csv('/content/drive/MyDrive/Datasets/full_data.csv')

# === Preprocesare dataset ===
stroke_encoded = stroke_set.copy()
stroke_encoded['gender'] = stroke_encoded['gender'].map({'Male':0, 'Female':1})
stroke_encoded['ever_married'] = stroke_encoded['ever_married'].map({'No':0, 'Yes':1})
stroke_encoded['Residence_type'] = stroke_encoded['Residence_type'].map({'Urban':0, 'Rural':1})
stroke_numeric = stroke_encoded[['age','avg_glucose_level','bmi','hypertension',
                                 'heart_disease','stroke','gender','ever_married','Residence_type']]

# === Împărțire în caracteristici și țintă ===
X = stroke_numeric.drop('stroke', axis=1)
y = stroke_numeric['stroke']

# === Normalizare pentru Elbow și CV ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# === Determinarea valorii optime a lui k ===
k_values = range(1, 31)
error_rates = []
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Elbow Method – eroare pe setul de test
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error_rates.append(1 - accuracy_score(y_test, y_pred))
    
    # Cross-Validation – scor mediu 5-fold
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Valori optime ale lui k
best_k_elbow = k_values[np.argmin(error_rates)]
best_k_cv = k_values[np.argmax(cv_scores)]

print(f"k optim (Elbow Method): {best_k_elbow}")
print(f"k optim (Cross-Validation): {best_k_cv}")

# === Grafic comparativ Elbow vs Cross-Validation ===
plt.figure(figsize=(10,6))
plt.plot(k_values, error_rates, marker='o', color='tab:green', label='Elbow Method (Test Error)')
plt.plot(k_values, [1 - x for x in cv_scores], marker='s', color='tab:orange', label='Cross-Validation (1 - Accuracy)')

# Marcăm valorile optime
plt.scatter(best_k_elbow, error_rates[best_k_elbow-1], color='tab:green', s=100, zorder=5)
plt.scatter(best_k_cv, 1 - cv_scores[best_k_cv-1], color='tab:orange', s=100, zorder=5)
plt.axvline(best_k_elbow, linestyle='--', color='tab:green', alpha=0.5)
plt.axvline(best_k_cv, linestyle='--', color='tab:orange', alpha=0.5)
plt.text(best_k_elbow+0.5, error_rates[best_k_elbow-1]+0.005, f'k={best_k_elbow}', color='tab:green')
plt.text(best_k_cv+0.5, 1 - cv_scores[best_k_cv-1]+0.005, f'k={best_k_cv}', color='tab:orange')

plt.title('Determinarea valorii optime a lui k: Elbow Method vs Cross-Validation', fontsize=13)
plt.xlabel('Numărul de vecini (k)')
plt.ylabel('Eroare / (1 - acuratețe)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# === Compararea timpului de execuție ===
scaler_std = StandardScaler()
X_scaled_std = scaler_std.fit_transform(X)
X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(X_scaled_std, y, test_size=0.2, random_state=42)

k_values_time = range(1, 26)
elbow_errors = []

start_elbow = time.time()
for k in k_values_time:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_std, y_train_std)
    score = knn.score(X_test_std, y_test_std)
    elbow_errors.append(1 - score)
end_elbow = time.time()
elbow_time = end_elbow - start_elbow

cv_errors = []
start_cv = time.time()
for k in k_values_time:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled_std, y, cv=5)
    cv_errors.append(1 - scores.mean())
end_cv = time.time()
cv_time = end_cv - start_cv

print(f"Timp de execuție Elbow Method: {elbow_time:.3f} secunde")
print(f"Timp de execuție Cross-Validation: {cv_time:.3f} secunde")

# Bar chart timp de procesare cu valori
plt.figure(figsize=(6,4))
bars = plt.bar(['Elbow Method', 'Cross-Validation'], [elbow_time, cv_time], color=['#15b01a', '#ff7f0e'], edgecolor='black', alpha=0.9)
plt.ylabel('Timp de execuție (secunde)')
plt.title('Compararea timpului de procesare între Elbow și Cross-Validation')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.05, f'{height:.3f} s', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
