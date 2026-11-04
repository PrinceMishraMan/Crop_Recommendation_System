from flask import Flask, request, render_template
import requests
import numpy as np
import joblib
from PIL import Image
import tensorflow.lite as tflite
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from RAG_Engine import generate_prescription
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


model = None
sc = None
interpreter = None
def load_models_lazily():
    global model, sc, cnn
    if interpreter is None:
        print("Loading models into memory for the first time...")
        model = joblib.load(open('models/model.joblib', 'rb'))
        sc = joblib.load(open('models/scaler.joblib', 'rb'))
        interpreter = tflite.Interpreter(model_path="models/plant_disease_model.tflite")
        interpreter.allocate_tensors()
        input_idx = interpreter.get_input_details()[0]['index']
        output_idx = interpreter.get_output_details()[0]['index']
        print("Models loaded successfully!")

class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
    'Tomato___healthy'
]

def parse_disease_prediction(raw_class: str):
    if "___" in raw_class:
        crop_part, disease_part = raw_class.split("___", 1)
    else:
        crop_part, disease_part = "Plant", raw_class

    crop_clean = crop_part.replace("_", " ").strip()
    disease_clean = disease_part.replace("_", " ").strip()
    return crop_clean, disease_clean


def get_live_weather(lat: str, lon: str) -> dict:
    weather = {"temp": 27.0, "humidity": 65, "condition": "Normal"}
    if lat and lon:
        try:
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation"
            res = requests.get(weather_url, timeout=3.5).json()
            weather["temp"] = res['current']['temperature_2m']
            weather["humidity"] = res['current']['relative_humidity_2m']
            weather["condition"] = "Rainy" if res['current']['precipitation'] > 0.1 else "Clear"
        except Exception as e:
            print(f"[-] Weather API fallback error: {e}")
    return weather

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
        if 'image' not in request.files or request.files['image'].filename == '':
            return render_template('disease_pred.html', error_msg="Please upload a valid image.")

        load_models_lazily()

        img_file = request.files['image']
        filename = secure_filename(img_file.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img_file.save(img_path)

        img = image.load_img(img_path, target_size=(128, 128))
        img_arr = image.img_to_array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)
        
        predictions = cnn.predict(img_arr)
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[0][predicted_idx])
        raw_predicted_class = CLASS_NAMES[predicted_idx]

        crop_name, disease_name = parse_disease_prediction(raw_predicted_class)

        lat = request.form.get('latitude')
        lon = request.form.get('longitude')
        language = request.form.get('language', 'English')
        weather_info = get_live_weather(lat, lon)

        advisory_text = None
        if "healthy" not in disease_name.lower():
            try:
                advisory_text = generate_prescription(
                    crop=crop_name,
                    disease=disease_name,
                    confidence=confidence,
                    weather=weather_info,
                    language=language
                )
            except Exception as e:
                print(f"[-] RAG Generation Error: {e}")
                advisory_text = "Unable to generate university advisory at this time."

        return render_template(
            'disease_pred.html',
            crop=crop_name,
            disease=disease_name,
            confidence=f"{confidence * 100:.1f}%",
            advisory=advisory_text,
            disease_text=f"{crop_name} - {disease_name}"
        )

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




