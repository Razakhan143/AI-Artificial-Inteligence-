from flask import Flask, request, render_template
import numpy as np
import pickle

# Load models
model = pickle.load(open('D:\\PROFESSIONAL\\AI\\projects\\DIABETES PREDICTION\\diabetes.pkl', 'rb'))
preprocessor = pickle.load(open('D:\\PROFESSIONAL\\AI\\projects\\DIABETES PREDICTION\\preprocessor.pkl', 'rb'))

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('app_front.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        gender = request.form.get('gender')
        age = request.form.get('age')
        hypertension = request.form.get('hypertension')
        heart_disease = request.form.get('heart_disease')
        smoking = request.form.get('smoking')
        bmi = request.form.get('bmi')
        gh = request.form.get('gh')
        bgl = request.form.get('bgl')

        # Check if all fields are present
        if None in [gender, age, hypertension, heart_disease, smoking, bmi, gh, bgl]:
            return render_template('app_front.html', prediction="Please provide all details.")

        # Convert form values to the format expected by the preprocessor
        gender_map = {'m': 'Male', 'f': 'Female'}
        hypertension_map = {'Y': 1, 'N': 0}
        heart_disease_map = {'Y': 1, 'N': 0}

        gender = gender_map.get(gender, gender)
        hypertension = hypertension_map.get(hypertension, hypertension)
        heart_disease = heart_disease_map.get(heart_disease, heart_disease)

        features = np.array([[gender, age, hypertension, heart_disease, smoking, bmi, gh, bgl]], dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = model.predict(transformed_features).reshape(1, -1)
        if prediction==1:
            return render_template('app_front.html', prediction="YOU ARE AT THE RISK OF DIABETES, PLEASE CONSULT")  
        else:
            return render_template('app_front.html', prediction="YOU ARE NOT AT THE RISK OF DIABETES.")

if __name__ == '__main__':
    app.run(debug=True)
