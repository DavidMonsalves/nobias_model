from flask import Flask, request, render_template
from flask_cors import CORS
import json
import numpy as np
import pickle

modelG = pickle.load(open('dt_model_PG_nobias.pkl', 'rb'))
modelAttribution = pickle.load(open('dt_model_PAttribution_nobias.pkl', 'rb'))
modelPerformance = pickle.load(open('dt_model_PPerformance_nobias.pkl', 'rb'))
modelUnconscious = pickle.load(open('dt_model_PUnconscious_nobias.pkl', 'rb'))
modelMaternal = pickle.load(open('dt_model_PMaternal_nobias.pkl', 'rb'))

app = Flask(__name__)
CORS(app)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predictG', methods = ['GET', 'POST'])
def home():
    if request.method == 'POST':
        event = json.loads(request.data)
        values = event['values']
        pre = np.array(values)
        pre = pre.reshape(1, -1)
        res = modelG.predict(pre)
        return str(res[0])

@app.route('/predictAttribution', methods = ['GET', 'POST'])
def home2():
    if request.method == 'POST':
        event = json.loads(request.data)
        values = event['values']
        pre = np.array(values)
        pre = pre.reshape(1, -1)
        res = modelAttribution.predict(pre)
        return str(res[0])

@app.route('/predictPerformance', methods = ['GET', 'POST'])
def home3():
    if request.method == 'POST':
        event = json.loads(request.data)
        values = event['values']
        pre = np.array(values)
        pre = pre.reshape(1, -1)
        res = modelPerformance.predict(pre)
        return str(res[0])

@app.route('/predictUnconscious', methods = ['GET', 'POST'])
def home4():
    if request.method == 'POST':
        event = json.loads(request.data)
        values = event['values']
        pre = np.array(values)
        pre = pre.reshape(1, -1)
        res = modelUnconscious.predict(pre)
        return str(res[0])

@app.route('/predictMaternal', methods = ['GET', 'POST'])
def home5():
    if request.method == 'POST':
        event = json.loads(request.data)
        values = event['values']
        pre = np.array(values)
        pre = pre.reshape(1, -1)
        res = modelMaternal.predict(pre)
        return str(res[0])

if __name__ == '__main__':
    app.run(debug=True)
