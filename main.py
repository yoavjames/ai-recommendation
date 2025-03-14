import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib  # To save the trained model
from flask import Flask, request, jsonify

# Sample Data (Replace with actual data)
data = {
    'Clinical_Condition': ['Acute Blood Loss', 'Thrombocytopenia', 'Coagulopathy', 'Chronic Anemia'],
    'PCV': [15, 40, 35, 20],
    'Platelets': [200000, 20000, 150000, 250000],
    'PT_aPTT': ['Normal', 'Normal', 'Prolonged', 'Normal'],
    'Total_Protein': [6.5, 6.8, 7.0, 5.5],
    'Recommended_Product': ['Whole Blood', 'Platelet-Rich Plasma', 'Fresh Frozen Plasma', 'Packed RBCs']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Encode Categorical Variables
le_condition = LabelEncoder()
le_pt_aptt = LabelEncoder()
le_product = LabelEncoder()

df['Clinical_Condition'] = le_condition.fit_transform(df['Clinical_Condition'])
df['PT_aPTT'] = le_pt_aptt.fit_transform(df['PT_aPTT'])
df['Recommended_Product'] = le_product.fit_transform(df['Recommended_Product'])

# Split Features and Labels
X = df.drop(columns=['Recommended_Product'])
y = df['Recommended_Product']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the Model and Encoders
joblib.dump(model, 'blood_product_model.pkl')
joblib.dump(le_condition, 'condition_encoder.pkl')
joblib.dump(le_pt_aptt, 'pt_aptt_encoder.pkl')
joblib.dump(le_product, 'product_encoder.pkl')

print("Model and encoders saved successfully!")

# Flask API for Model Deployment
app = Flask(__name__)

# Load the saved model and encoders
model = joblib.load('blood_product_model.pkl')
le_condition = joblib.load('condition_encoder.pkl')
le_pt_aptt = joblib.load('pt_aptt_encoder.pkl')
le_product = joblib.load('product_encoder.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    clinical_condition = le_condition.transform([data['Clinical_Condition']])[0]
    pt_aptt = le_pt_aptt.transform([data['PT_aPTT']])[0]

    features = np.array([[clinical_condition, data['PCV'], data['Platelets'], pt_aptt, data['Total_Protein']]])
    prediction = model.predict(features)[0]
    recommended_product = le_product.inverse_transform([prediction])[0]

    return jsonify({'Recommended_Product': recommended_product})


@app.route('/', methods=['GET'])
def hello():
    return {'Hello': 'World'}


if __name__ == '__main__':
    from os import environ
    app.run(host='0.0.0.0', port=environ.get('PORT', 5000), debug=True)