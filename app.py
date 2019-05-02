import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from flask import Flask,request,render_template


app = Flask(__name__)

@app.route('/')

def index():

    return render_template('index.html')

if __name__ == '__main':
    app.run()
