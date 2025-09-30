import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import joblib

# Load dataset
data = pd.read_csv('symptoms.csv')

# Target variable should be Outcome Variable (Positive/Negative)
target = data['Outcome Variable']
data.drop(['Disease', 'Outcome Variable'], axis=1, inplace=True)

# Encode categorical features
oe = OneHotEncoder()
data = oe.fit_transform(data).toarray()

# Encode target
le = LabelEncoder()
target = le.fit_transform(target)   # Positive=1, Negative=0

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

# Train model (Random Forest)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)

# Predictions
y_predict = rf.predict(x_test)

# Metrics
print("Accuracy Score:", accuracy_score(y_test, y_predict))
print("Recall Score:", recall_score(y_test, y_predict, average="binary"))
print("Precision Score:", precision_score(y_test, y_predict, average="binary"))

joblib.dump(rf, 'DiseasePredictModel.joblib')
joblib.dump(oe, 'encoder.joblib')
joblib.dump(le, 'label_encoder.joblib')