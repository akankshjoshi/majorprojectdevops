# train_model.py
import pandas as pd
import gzip
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

df = pd.read_csv("../data/ipl_data.csv")
df = df.drop(columns=['mid', 'date', 'venue', 'batsman', 'bowler', 'striker', 'non-striker'])
df = pd.get_dummies(df, columns=['bat_team', 'bowl_team'])

X = df.drop('total', axis=1)
y = df['total']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Save both model and columns
joblib.dump(model, '../front/score_predictor.pkl')
joblib.dump(X.columns, '../front/model_columns.pkl')

with gzip.open('../front/score_predictor.pkl.gz', 'wb') as f:
    joblib.dump(model, f)

print("✅ Model trained and saved.")

