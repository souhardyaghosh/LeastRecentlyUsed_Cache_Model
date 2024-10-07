import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from Excel
df = pd.read_excel(r'C:\Users\tosou\OneDrive\Desktop\CAO\lru_cache_data.xlsx')

# Generate a synthetic Eviction column (0 or 1)
np.random.seed(42)
df['Eviction'] = np.random.randint(0, 2, len(df))

# Features: All columns except 'Eviction' and 'Timestamp'
X = df.drop(['Eviction', 'Timestamp'], axis=1)
y = df['Eviction']

# Normalize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for KNN using GridSearchCV
param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11]}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best model from grid search
knn_best = grid_search.best_estimator_

# Predict on the test set using the best model
y_pred = knn_best.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print accuracy and classification report
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(report)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Evicted', 'Evicted'], yticklabels=['Not Evicted', 'Evicted'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature Importance using Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_
feature_names = X.columns

# Create a DataFrame for feature importance
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.show()

# Set a threshold for eviction based on predicted probabilities
eviction_probabilities = knn_best.predict_proba(X_test)[:, 1]  # Probability of eviction (class 1)
threshold = 0.5  # You can tune this value
eviction_decisions = (eviction_probabilities > threshold).astype(int)

# Add eviction decisions to the DataFrame
df['Eviction_Decision'] = 0  # Initialize the column
df.loc[y_test.index, 'Eviction_Decision'] = eviction_decisions  # Assign predictions

# Analyze caches that are marked for eviction
evicted_caches = df[df['Eviction_Decision'] == 1]
print(f"Total caches marked for eviction: {len(evicted_caches)}")
print(evicted_caches[['ItemID', 'Eviction_Decision'] + list(X.columns)])

# Save evicted caches to a new Excel file
evicted_caches.to_excel('evicted_caches.xlsx', index=False)

# Visualize ROC Curve
RocCurveDisplay.from_estimator(knn_best, X_test, y_test)
plt.title('ROC Curve for KNN')
plt.show()
