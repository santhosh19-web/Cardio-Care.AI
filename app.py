from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Collect and scale data
    input_values = [float(x) for x in request.form.values()]
    scaled_data = scaler.transform([input_values])
    
    # 2. Predict
    prediction = model.predict(scaled_data)
    
    # 3. Formulate Output
    if prediction[0] == 1:
        text = "⚠️ Assessment: High risk of heart disease detected. Please consult a specialist."
        res_class = "danger"
    else:
        text = "✅ Assessment: Your clinical data indicates a low risk of heart disease."
        res_class = "success"
        
    return render_template('index.html', prediction_text=text, res_class=res_class)

if __name__ == "__main__":
    app.run(debug=True)