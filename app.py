# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 10:04:32 2021

@author: mayan 
"""

from flask import Flask, render_template, request
import math
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

dic = {0 : 'NEGATIVE', 1 : 'POSITIVE'}

model = load_model('CNN_model_covid.h5')

model.make_predict_function() 

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(224,224))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 224,224,3) 
	p = model.predict(i)
	return p[0][0]*100
print('Model loaded. Check http://127.0.0.1:5000/')
# routes
@app.route("/", methods=['GET']) 
def main():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path) 

	return render_template("index.html", prediction = p, img_path = img_path , per=dic[math.floor(p/50)] )

if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)