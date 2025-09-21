from flask import Flask, request, render_template
import numpy as np
import pickle
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    val = np.array(int_features).reshape(1, -1)
    scaled_val = sc.transform(val)
    prediction = model.predict(scaled_val)
    output = prediction[0]
    return render_template('index.html', prediction_text='Recommended Crop is: {}'.format(output))

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


