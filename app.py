from flask import Flask, request, render_template
import json
import numpy as np
import pickle

model = pickle.load(open('dt_model_nobias.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods = ['GET', 'POST'])
def home():
    if request.method == 'POST':
        event = json.loads(request.data)
        values = event['values']
        pre = np.array(values)
        pre = pre.reshape(1, -1)
        res = model.predict(pre)
        return str(res[0])

if __name__ == '__main__':
    app.run(debug=True)
