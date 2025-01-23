from flask import Flask, request, render_template
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the machine learning model
model = joblib.load("ML PROJECTS\life expectancy\ExtraTreesRegressor.joblib")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input data from form
        data = [
            request.form['Year'],
            request.form['Status'],
            request.form['AdultMortality'],
            request.form['Alcohol'],
            request.form['percentageexpenditure'],
            request.form['HepatitisB'],
            request.form['Measles'],
            request.form['under_five_deaths'],
            request.form['Polio'],
            request.form['Totalexpenditure'],
            request.form['Diphtheria'],
            request.form['HIV_AIDS'],
            request.form['GDP'],
            request.form['Population'],
            request.form['thinness_1_19_years'],
            request.form['Incomecompositionofresources'],
            request.form['Schooling'],
            request.form['Continent']
        ]

        # Preprocess the data (convert to appropriate types if necessary)
        data = np.array(data).reshape(1, -1)

        # Predict using the loaded model
        prediction = model.predict(data)

        return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)