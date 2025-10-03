from flask import Flask,request,jsonify,render_template
import joblib
import numpy as np
model = joblib.load('planetModel.joblib')
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form.get('dec')),
            float(request.form.get('st_pmdec')),
            float(request.form.get('pl_tranmid')),
            float(request.form.get('pl_tranmiderr1')),
            float(request.form.get('pl_orbper')),
            float(request.form.get('pl_orbpererr1')),
            float(request.form.get('pl_trandurh')),
            float(request.form.get('pl_trandurherr1')),
            float(request.form.get('pl_trandep')),
            float(request.form.get('pl_trandeperr1')),
            float(request.form.get('pl_rade')),
            float(request.form.get('pl_radeerr1')),
            float(request.form.get('pl_insol')),
            float(request.form.get('pl_eqt')),
            float(request.form.get('st_tmag')),
            float(request.form.get('st_dist')),
            float(request.form.get('st_disterr1')),
            float(request.form.get('st_tefferr1'))
        ]
        input_data = np.array([features])  # shape: (1, 18)
        prediction = model.predict(input_data)
        return jsonify({'prediction': str(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})    
    
if __name__ == '__main__':
    app.run(debug=True)    