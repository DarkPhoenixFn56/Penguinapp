import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv('penguins_cleaned.csv')

# Separate features and target
X = df.drop('species', axis=1)
y = df['species']

# One-hot encode categorical columns
X = pd.get_dummies(X)

# Train model
clf = RandomForestClassifier()
clf.fit(X, y)

# Save model
with open('penguins_clf.pkl', 'wb') as f:
    pickle.dump(clf, f)
