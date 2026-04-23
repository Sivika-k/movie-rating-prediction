import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv('IMDb Movies India.csv', encoding='latin1')

# Clean column names
df.columns = df.columns.str.strip()

# Drop rows without rating (target)
df = df.dropna(subset=['Rating'])

# -------------------------------
# DATA CLEANING
# -------------------------------

# Fill missing values
df['Genre'] = df['Genre'].fillna('Unknown')
df['Director'] = df['Director'].fillna('Unknown')
df['Actor 1'] = df['Actor 1'].fillna('Unknown')

# Convert Votes
df['Votes'] = df['Votes'].astype(str).str.replace(',', '')
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce').fillna(0)

# Extract Year
df['Year'] = df['Year'].str.extract('(\d{4})')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Convert Duration
df['Duration'] = df['Duration'].str.replace(' min', '')
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

# Fill remaining missing values
df[['Year', 'Duration']] = df[['Year', 'Duration']].fillna(0)

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------

# Use only Genre for encoding (avoid high cardinality problem)
df = pd.get_dummies(df, columns=['Genre'], drop_first=True)

# Features & Target
X = df.drop(columns=['Name', 'Rating', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'])
y = df['Rating']

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# MODEL
# -------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# PREDICTION
# -------------------------------

y_pred = model.predict(X_test)

# -------------------------------
# EVALUATION
# -------------------------------

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# -------------------------------
# SAMPLE PREDICTION
# -------------------------------

sample = X_test.iloc[0:1]

print("\nSample Input:\n", sample)

predicted_rating = model.predict(sample)
print("\nPredicted Rating:", predicted_rating[0])
print("Actual Rating:", y_test.iloc[0])