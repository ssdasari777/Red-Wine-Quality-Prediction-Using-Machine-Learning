
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("model (3).pkl", "rb"))

@app.route('/')
def index():
    return render_template('about.html')

@app.route('/formsg')
def formsg():
    return render_template('formmsg.html')


@app.route('/predict', methods= ["POST", "GET"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    result = model.predict(features)
    return render_template('submit.html', result = result) 



if __name__ == '__main__':
    app.run(debug=True)
