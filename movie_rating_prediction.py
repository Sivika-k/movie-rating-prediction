import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv('IMDb Movies India.csv', encoding='latin1')

df.columns = df.columns.str.strip()
df = df.dropna(subset=['Rating'])

# Clean data
df['Genre'] = df['Genre'].fillna('Unknown')
df['Votes'] = df['Votes'].astype(str).str.replace(',', '')
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce').fillna(0)

df['Year'] = df['Year'].str.extract('(\d{4})')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0)

df['Duration'] = df['Duration'].str.replace(' min', '')
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce').fillna(0)

# Feature engineering
df = pd.get_dummies(df, columns=['Genre'], drop_first=True)

X = df.drop(columns=['Name','Rating','Director','Actor 1','Actor 2','Actor 3'])
y = df['Rating']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Sample prediction
sample = X_test.iloc[0:1]
print("Prediction:", model.predict(sample))
print("Actual:", y_test.iloc[0])
