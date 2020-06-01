from flask import Flask,request
import pandas as pd 
import numpy as np
import pickle
import flasgger
from flasgger import Swagger 

app = Flask(__name__)
Swagger(app)

pickle_in = open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
	return "welcome to the test API"

@app.route('/getPrediction',methods = ['GET'])
def predict_note():

	"""Bank note authentication
	---
	parameters:
		- name: variance 
		  in: query
		  type: number
		  required: true
		- name: skewness
		  in: query
		  type: number
		  required: true
		- name: curtosis 
		  in: query
		  type: number
		  required : true
		- name: entropy 
		  in: query
		  type: number
		  required: true
	responses:
		200 :
			description: The output values
	"""
	# content = request.get_json()
	variance = request.args.get('variance')
	skewness = request.args.get('skewness')
	curtosis = request.args.get('curtosis')
	entropy =  request.args.get('entropy')

	prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
	# print('oyee'+prediction)
	# prediction = classifier.predict([[content['variance'],content['skewness'],content['curtosis'],content['entropy']]])

	fun = lambda x : 'True' if x == 1 else 'False'
	return 'The prediction value is '+ fun(prediction)


@app.route('/getPredictionFile',methods = ['POST'])
def predict_file():

	"""Bank note authentication using 
	   test file
	---
	parameters:
		- name: file
		  in: formData
		  type: file
		  required: true

	responses:
		200:
			description: The output values

	"""
	
	df_test = pd.read_csv(request.files.get('file')) 
	prediction = classifier.predict(df_test)
	return 'prediction values for the csv is '+ str(list(prediction))

if __name__ == '__main__':
	app.run(host='127.0.0.1')

