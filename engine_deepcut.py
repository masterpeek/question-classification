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
from pythainlp.tokenize import word_tokenize

def deepCut(text):
    word_cut = word_tokenize(text, engine="deepcut")
    return word_cut

app = Flask(__name__)

@app.route('/api/v1.0/deepcut/', methods=['POST'])
def broker():
    if not request.json or not 'text' in request.json:
        abort(400)
    
    getText = request.json['text']

    result_deepcut = deepCut(getText)

    return jsonify({'result_deepcut': result_deepcut})

if __name__ == '__main__':
    app.run(debug=True, port = 8005)