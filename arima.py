import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from Excel
df = pd.read_excel(r'C:\Users\tosou\OneDrive\Desktop\CAO\lru_cache_data.xlsx')

# Generate a synthetic Eviction column (0 or 1)
np.random.seed(42)
df['Eviction'] = np.random.randint(0, 2, len(df))

# Features for KNN
X = df.drop(['Eviction', 'Timestamp'], axis=1)
y = df['Eviction']

# Normalize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# KNN Predictions
y_pred_knn = knn.predict(X_test)

# Evaluate KNN Performance
accuracy_knn = accuracy_score(y_test, y_pred_knn)
report_knn = classification_report(y_test, y_pred_knn)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)

# Print KNN results
print(f"KNN Accuracy: {accuracy_knn * 100:.2f}%")
print("KNN Classification Report:")
print(report_knn)

# Prepare data for ARIMA
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

# Resample the data to daily frequency, using mean for aggregation
arima_data = df.resample('D').mean().fillna(method='ffill')

# Ensure Eviction column exists
if 'Eviction' not in arima_data.columns:
    print("Eviction column not found in resampled data.")
else:
    print("Eviction column found.")

# Fit ARIMA model
model = ARIMA(arima_data['Eviction'], order=(1, 1, 1))
model_fit = model.fit()

# Make predictions
arima_forecast = model_fit.forecast(steps=len(y_test))
arima_forecast_binary = (arima_forecast > 0.5).astype(int)  # Convert to binary (0 or 1)

# Save ARIMA results to Excel
arima_results = pd.DataFrame({
    'Actual_Eviction': y_test.values,
    'Predicted_Eviction': arima_forecast_binary
})

# Print ARIMA results
accuracy_arima = accuracy_score(y_test, arima_forecast_binary)
print(f"ARIMA Accuracy: {accuracy_arima * 100:.2f}%")

# Check actual vs predicted values
print("Actual vs Predicted Evictions:")
print(arima_results)

# Generate graphs for KNN and ARIMA predictions
plt.figure(figsize=(12, 6))

# KNN Results
plt.subplot(1, 2, 1)
plt.plot(y_test.index, y_test, label='Actual Eviction', color='blue', marker='o', markersize=3)
plt.plot(y_test.index, y_pred_knn, label='Predicted Eviction (KNN)', color='orange', marker='x', markersize=3)
plt.title('Actual vs Predicted Eviction (KNN)')
plt.xlabel('Date')
plt.ylabel('Eviction')
plt.legend()
plt.grid()

# ARIMA Results
plt.subplot(1, 2, 2)
plt.plot(y_test.index, y_test, label='Actual Eviction', color='blue', marker='o', markersize=3)
plt.plot(y_test.index, arima_forecast_binary, label='Predicted Eviction (ARIMA)', color='orange', marker='x', markersize=3)
plt.title('Actual vs Predicted Eviction (ARIMA)')
plt.xlabel('Date')
plt.ylabel('Eviction')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('comparison_predictions_graph.png')  # Save the comparison figure
plt.show()

# Print confusion matrix for ARIMA
conf_matrix_arima = confusion_matrix(y_test, arima_forecast_binary)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_arima, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Evicted', 'Evicted'], 
            yticklabels=['Not Evicted', 'Evicted'])
plt.title('Confusion Matrix for ARIMA')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
