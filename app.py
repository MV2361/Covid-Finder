


from flask import Flask, render_template, request
import math
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

dic = {0 : 'NEGATIVE', 1 : 'POSITIVE'}

CNN=load_model('CNN_model_covid.h5')
RNET=load_model('ResNet_model_covid.h5')
GN=load_model('GoogleNet_final_covid_latest.h5')

members=[CNN, RNET, GN]

#members.make_predict_function() 

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(224,224))
    i = image.img_to_array(i)/255.0
    i = i.reshape(1, 224,224,3)
    Ans=0
    for model in members: 
        model.make_predict_function() 
        p = model.predict(i)
        if(model == GN):
            flag=0
            if(p[0][0]>=0.5):
                flag+=1
            if(p[1][0]>=0.5):
                flag+=1
            if(p[2][0]>=0.5):
                flag+=1
            if(flag>=2):
                Ans+=1
        else:
            if(p[0][0]>0.5):
                Ans+=1
    if(Ans>=2):
        return "POSITIVE"
    else:
        return "NEGATIVE"
	
#print('Model loaded. Check http://127.0.0.1:5000/')
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

	return render_template("index.html", prediction = p, img_path = img_path  )

if __name__ =='__main__':
	#app.debug = True
	app.run(host='0.0.0.0' , port=8080)