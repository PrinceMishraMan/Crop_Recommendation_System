from flask import Flask, request, render_template
import requests
import numpy as np
import pickle
import joblib
import tensorflow as tf
from keras.preprocessing import image
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
sc = joblib.load(open('scaler.joblib', 'rb'))
cnn = tf.keras.models.load_model('my_cnn_model_2.keras')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        nitrogen = request.form.get('nitrogen')
        phosphorus = request.form.get('phosphorus')
        potassium = request.form.get('potassium')
        lat = request.form.get('latitude')
        lon = request.form.get('longitude')
        temperature = None
        humidity = None
        ph = request.form.get('ph')
        rainfall = None
        if lat and lon:
            try:
                weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation"
                weather_response = requests.get(weather_url).json()
                
                temperature = weather_response['current']['temperature_2m']
                humidity = weather_response['current']['relative_humidity_2m']
                rainfall = weather_response['current']['precipitation']
            except Exception as e:
                print(f"Error fetching from Open-Meteo API: {e}")
        
        if temperature is None or humidity is None or rainfall is None:
            temperature = request.form.get('temperature')
            humidity = request.form.get('humidity')
            rainfall = request.form.get('rainfall')
            
            if not temperature or not humidity or not rainfall:
                return render_template('index.html', prediction_text="Error: Missing weather/rainfall data. Please allow location access or type it in manually.")
        
        int_features = [
            float(nitrogen), 
            float(phosphorus), 
            float(potassium), 
            float(temperature),
            float(humidity),
            float(ph), 
            float(rainfall)
        ]
        val = np.array(int_features).reshape(1, -1)
        scaled_val = sc.transform(val)
        prediction = model.predict(scaled_val)
        output = prediction[0]
        return render_template('index.html', prediction_text='Recommended Crop is: {}'.format(output))
    return render_template('index.html')

@app.route('/disease', methods=['GET', 'POST'])
def disease():
    if request.method == 'POST':
        img = request.files['image']
        img.save(img.filename)
        img = image.load_img(img.filename, target_size = (128, 128))
        img = image.img_to_array(img)
        img = img/255.0
        img = np.expand_dims(img, axis = 0)
        result = cnn.predict(img)
        class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                   'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                   'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                   'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                   'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
                   'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
                   'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
                   'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
        predicted_class = class_names[np.argmax(result)]
        print(predicted_class)
        return render_template('disease_pred.html', disease_text='Predicted Disease is: {}'.format(predicted_class))
    return render_template('disease_pred.html')

@app.route('/crop_pred')
def crop():
    return render_template('crop_pred.html')

@app.route('/crop_pest')
def pest():
    return render_template('pest_pred.html')


@app.route('/login')
def login():
    return render_template('login.html')


if __name__ == "__main__":

    app.run(debug=True)






