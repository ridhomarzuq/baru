import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data_path = 'heart_2020_cleaned1_final.csv'
data = pd.read_csv(data_path)

# Assuming 'HeartDisease' is the target variable and the rest are features
X = data.drop(columns=['HeartDisease'])
y = data['HeartDisease']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
model_path = 'random_forest_model.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(model, file)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
