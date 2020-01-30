from flask import Flask, jsonify
from flask import make_response
from flask import request
from flask import abort
from flask import json

import requests
import json

import tensorflow as tf
import os
import time
import sys
import requests
from io import BytesIO

import numpy as np
from PIL import Image

from pickle import dump
from keras.models import load_model
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from pythainlp.tokenize import word_tokenize

def preprocessText(input_text):
    word = input_text
    headers = {'content-type': 'application/json'}
    URL = 'http://127.0.0.1:8005/api/v1.0/deepcut/'
    data = {'text': word}
    res = requests.post(URL, data = json.dumps(data), headers=headers)
    result_deepcut = res.json().get('result_deepcut')
    return result_deepcut

classNameCat = { 0:'anatomy', 1:'social', 2:'financial', 3:'work', 4:'education', 5:'lifestyle', 6:'channel', 7:'other' }

def load_dataset(fileName):
    return load(open(fileName, 'rb'))

trainX, trainYCat = load_dataset('event_order_form_fix.pkl')

def max_length(lines):
    return max([len(s.split()) for s in lines])

length = max_length(trainX)

def encode_text(tokenizer, lines, length):
    encoded = tokenizer.texts_to_sequences(lines)
    padded = pad_sequences(encoded, maxlen = length, padding='post')
    return padded

def loadModel():
    global graph
    global predict_model

    predict_model = load_model('textCatQuestionModelFix.h5')

    graph = tf.get_default_graph()


def predict(text):
    global predict_model
    global graph
    with graph.as_default():
        tokenizerName = 'tokenizerCat_fix.pkl'

        tokenizer = load_dataset(tokenizerName)

        t = encode_text(tokenizer, [text], length)

        classNameCat = { 0:'anatomy', 1:'social', 2:'financial', 3:'work', 4:'education', 5:'lifestyle', 6:'channel', 7:'other' }

        pred = predict_model.predict([t, t, t])

        res = np.argmax(pred, axis=1)

        classification = classNameCat[res[0]]
        accuracy_convert = pred[0][res[0]]
        accuracy = str(accuracy_convert)

        print('encode :', t)
        print('text :', text)
        print('lenght :', length)
        print('class :', classNameCat[res[0]])
        print('accuracy :', pred[0][res[0]])

        return classification, accuracy

app = Flask(__name__)

@app.route('/api/v1.0/classification/', methods=['POST'])
def classification():
    if not request.json or not 'text' in request.json:
        abort(400)
        
    getText = request.json['text']

    text = preprocessText(getText)

    classification, accuracy = predict(text)

    print('classification :', classification)
    print('accuracy', accuracy)

    return jsonify({'classification': classification, 'accuracy': accuracy})
    
if __name__ == "__main__":
    loadModel()
    app.run(debug=True, port = 8000)