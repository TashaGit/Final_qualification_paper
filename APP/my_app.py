from flask import Flask, request, render_template

import tensorflow as tf
import pickle

app = Flask(__name__)


@app.route('/')
def choose_prediction_method():
    return render_template('main.html')


def nn_prediction(params):
    model = tf.keras.models.load_model('vkr_nn_model')
    pred = model.predict([params])
    return pred


def lr_prediction(params):
    with open('lr1_model.pkl', 'rb') as f:
        model = pickle.load(f)
        f.close()
    pred = model.predict([params])
    return pred


@app.route('/mn/', methods=['POST', 'GET'])
def nn_predict():
    message = ''
    if request.method == 'POST':
        param_list = ('plot', 'mup', 'ko', 'seg', 'tv', 'pp', 'mup', 'pr', 'ps', 'yn', 'shn', 'pln')
        params = []
        for i in param_list:
            param = request.form.get(i)
            params.append(param)
        params = [float(i.replace(',', '.')) for i in params]

        message = f'Прогноз значения Соотношение матрица-наполнитель для введенных параметров: {nn_prediction(params)}'
    return render_template('mn.html', message=message)


@app.route('/upr/', methods=['POST', 'GET'])
def lr_predict():
    message = ''
    if request.method == 'POST':
        param_list = ('mn', 'plot', 'mup', 'ko', 'seg', 'tv', 'pp', 'pr', 'ps', 'yn', 'shn', 'pln')
        params = []
        for i in param_list:
            param = request.form.get(i)
            params.append(param)
        params = [float(i.replace(',', '.')) for i in params]

        message = f'Прогноз значения Модуля упругости при растяжении для введенных параметров: {lr_prediction(params)} ГПа'
    return render_template('upr.html', message=message)

if __name__ == '__main__':
    app.run()
