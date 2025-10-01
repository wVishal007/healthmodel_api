import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import joblib

# ------------------------
# 1. Load dataset
# ------------------------
data = pd.read_csv("symptoms.csv")

# Target = Disease (not Outcome Variable anymore)
target = data["Disease"]

# Drop columns not used for features
data.drop(["Disease", "Outcome Variable"], axis=1, inplace=True)

# ------------------------
# 2. Separate categorical & numeric
# ------------------------
categorical_cols = ["Fever", "Cough", "Fatigue", "Difficulty Breathing",
                    "Gender", "Blood Pressure", "Cholesterol Level"]
numeric_cols = ["Age"]

# One-hot encode categorical
oe = OneHotEncoder(handle_unknown="ignore")
encoded_cats = oe.fit_transform(data[categorical_cols]).toarray()

# Keep numeric as-is
numerics = data[numeric_cols].values

# Final feature matrix
X = np.concatenate([encoded_cats, numerics], axis=1)

# Encode target (disease names → integers)
le = LabelEncoder()
y = le.fit_transform(target)

# ------------------------
# 3. Train-test split
# ------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ------------------------
# 4. Train Random Forest
# ------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(x_train, y_train)

# ------------------------
# 5. Evaluate
# ------------------------
y_predict = rf.predict(x_test)

print("✅ Model Evaluation:")
print("Accuracy Score:", accuracy_score(y_test, y_predict))
print("\nClassification Report:\n",
      classification_report(
          y_test,
          y_predict,
          labels=range(len(le.classes_)),   # all classes
          target_names=le.classes_
      )
)


# ------------------------
# 6. Save model + encoders
# ------------------------
joblib.dump(rf, "DiseasePredictModel.joblib")
joblib.dump(oe, "encoder.joblib")
joblib.dump(le, "label_encoder.joblib")

print("🎉 Model + encoders saved successfully!")
