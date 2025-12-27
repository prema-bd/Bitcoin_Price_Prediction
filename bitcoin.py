# =======================
# Import libraries
# =======================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

# =======================
# Load dataset
# =======================
df = pd.read_csv('bitcoin.csv')

# =======================
# Feature Engineering: Date & Quarter
# =======================
# Split 'Date' into year, month, day
splitted = df['Date'].str.split('-', expand=True)
df['year'] = splitted[0].astype('int')
df['month'] = splitted[1].astype('int')
df['day'] = splitted[2].astype('int')

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Add feature: is_quarter_end (1 if month is 3,6,9,12)
df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)

# Price difference features
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']

print("First 5 rows with features:")
print(df.head())

# =======================
# Exploratory Data Analysis
# =======================
print("\nShape of dataset:", df.shape)
print("\nStatistical Summary:")
print(df.describe())
print("\nDataset Info:")
df.info()

# Plot Bitcoin Close price
plt.figure(figsize=(15, 5))
plt.plot(df['Date'], df['Close'])
plt.title('Bitcoin Close Price', fontsize=15)
plt.xlabel('Date')
plt.ylabel('Price in USD')
plt.show()

# Check if Close and Adj Close are same
same_close = df[df['Close'] == df['Adj Close']].shape[0]
print("\nNumber of rows where Close == Adj Close:", same_close)

# Feature distributions
features = ['Open', 'High', 'Low', 'Close']
plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i+1)
    sn.histplot(df[col], kde=True)
    plt.title(f'{col} Distribution')
plt.tight_layout()
plt.show()

# Boxplots
plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i+1)
    sn.boxplot(df[col], orient='h')
    plt.title(f'{col} Boxplot')
plt.tight_layout()
plt.show()

# =======================
# Yearly Average Bar Plots
# =======================
data_grouped = df.groupby('year').mean()
plt.subplots(figsize=(20, 10))
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
    plt.subplot(2, 2, i+1)
    data_grouped[col].plot.bar()
    plt.title(f'Average {col} per Year')
plt.tight_layout()
plt.show()

# =======================
# Target Creation
# =======================
df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
df = df[:-1]  # drop last row with NaN target

# Pie chart of target distribution
plt.figure(figsize=(6,6))
plt.pie(df['Target'].value_counts().values, labels=[0, 1], autopct='%1.1f%%', colors=['lightcoral','lightskyblue'])
plt.title('Target Distribution (Down vs Up)')
plt.show()

# =======================
# Features and Target
# =======================
X = df[['Open', 'High', 'Low', 'Close', 'year', 'month', 'day', 'is_quarter_end', 'open-close', 'low-high']]
y = df['Target']

# Train-test split (time series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =======================
# Model Training and Evaluation
# =======================
models = {
    "Logistic Regression": LogisticRegression(),
    "Support Vector Classifier": SVC(),
    "XGBoost Classifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred))

# =======================
# Correlation Heatmap
# =======================
corr = df.corr()
plt.figure(figsize=(12, 10))
sn.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title("Feature Correlation Matrix")
plt.show()
