from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
with open('model (1).pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize the scaler (assuming you have the scaler as part of the model training)
  # This should be loaded or fitted using the same scaler from the training pipeline

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    scaler = StandardScaler()
    if request.method == 'POST':
        # Get input values from the form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Create input array for prediction
        input_data = [[N, P, K, temperature, humidity, ph, rainfall]]

        # Scale input data
        scaled_input = scaler.fit_transform(input_data)
        # return scaled_input

        # Make prediction
        prediction = model.predict(scaled_input)[0]
            
            
        label_data = {'labels' : ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
            'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
            'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'orange',
            'papaya', 'coconut', 'cotton', 'jute', 'coffee', 'groundnuts'],
            
            'labels_encoded' :[20, 11,  2,  9, 18, 13, 14,  1, 10, 19,  0, 12,  6, 21, 15, 16, 17,
        3,  5,  8,  4,  7]}


        df_label = pd.DataFrame(label_data)

        # label_mapping = dict(zip(df_label['labels'], df_label['labels_encoded']))
        predicted_crop = df_label[df_label['labels_encoded']==prediction]['labels'].values[0]

        # Render the result template with the prediction
        return render_template('result.html', prediction=predicted_crop)

if __name__ == '__main__':
    app.run(port=8000)

