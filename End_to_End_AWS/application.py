from flask import Flask,jsonify,render_template,request
import pandas as pd
import numpy as np
import pickle
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
   
    if request.method == 'POST':
        
        model=pickle.load(open('models/model.pkl','rb'))
        scaler=pickle.load(open('models/scaler.pkl','rb'))
        
        Temperature=float(request.form['Temperature'])
        RH=float(request.form['RH'])	
        Ws=float(request.form['Ws'])
        Rain=float(request.form['Rain'])
        FFMC=float(request.form['FFMC'])
        DMC=float(request.form['DMC'])
        ISI=float(request.form['ISI'])
        Classes	=float(request.form['Classes'])
        Region=float(request.form['Region'])


        data = np.array([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])

        data=scaler.transform(data)
        prediction=model.predict(data)
        return render_template('predict.html',prediction=prediction[0])
   
   
    elif request.method == 'GET':
        return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True)