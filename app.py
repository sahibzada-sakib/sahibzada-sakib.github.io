from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the machine learning model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        annual_income = float(request.form['annual_income'])
        applicant_age = int(request.form['applicant_age'])
        work_experience = int(request.form['work_experience'])
        marital_status = request.form['marital_status']
        house_ownership = request.form['house_ownership']
        vehicle_ownership = request.form['vehicle_ownership']
        
        # Preprocess user input
        # (you'll need to preprocess the input data similar to how you did it in your notebook)
        
        # Make prediction
        prediction = model.predict(np.array([[annual_income, applicant_age, work_experience, marital_status, house_ownership, vehicle_ownership]]))
        
        # Return prediction result to the HTML template
        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
