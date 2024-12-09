from flask import Flask,request,render_template
import numpy as np
import sklearn
import pickle

#loading models
dtr = pickle.load(open('ML PROJECTS\CROP_YIELD_PREDICTION\project_dtr.pkl','rb'))
preprocessor = pickle.load(open('ML PROJECTS\CROP_YIELD_PREDICTION\project_preprocessor.pkl','rb'))
#creating flask app
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('project_web.html')
@app.route("/predict",methods=['POST'])
def predict():
    if request.method=='POST':
        district= request.form['district']
        label= request.form['label']
        pesticides= request.form['pesticides']
        temperature= request.form['temperature']
        humidity= request.form['humidity']
        rainfall= request.form['rainfall']

    features = np.array([[district,	label,	pesticides,	temperature,	humidity,	rainfall]], dtype=object)

    transformed_features = preprocessor.transform(features)

    prediction = dtr.predict(transformed_features).reshape(1, -1)

    return render_template('project_web.html',prediction=prediction)


#python main
if __name__=='__main__':
    app.run(debug=True)