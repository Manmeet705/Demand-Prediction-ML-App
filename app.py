from flask import Flask, request, render_template
import joblib
import numpy as np

# Load the pre-trained model (ensure 'model.pkl' exists in the same directory)
model = joblib.load('pipeline_model.pkl')

app = Flask(__name__)

@app.route('/')
def Home():
    return render_template("index.html")

@app.route('/predict', methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == 'POST':
        # Retrieve form values and convert to floats
        features = [float(x) for x in request.form.values()]
        #f1 = float(request.form['feature1'])
        #f2 = float(request.form['feature2'])
        #f3 = float(request.form['feature3'])
        # Create feature array for prediction
        X_new = [np.array(features)]
        # Predict using the loaded model
        prediction = model.predict(X_new)
        # Render template from separate HTML file in 'templates' folder
        return render_template('SalesDemandPrediction.html', prediction=int(prediction[0]))
    else:
        return render_template('SalesDemandPrediction.html', prediction='')

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5000, debug=True)
    app.run(debug=True)
