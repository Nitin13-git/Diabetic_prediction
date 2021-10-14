#import relevant libraries for flask, html rendering and loading the ML model

from flask import Flask,request, url_for, redirect, render_template
import pickle
import pandas as pd

app = Flask(__name__)

#loading the SVM model and the preprocessor
model = pickle.load(open("svm_model.pkl", "rb"))
std = pickle.load(open('std.pkl','rb'))


#Index.html will be returned for the input
@app.route('/')
def hello_world():
    return render_template("index.html")


#predict function, POST method to take in inputs
@app.route('/predict',methods=['POST','GET'])
def predict():

    #take inputs for all the attributes through the HTML form
    pregnancies = request.form['1']
    glucose = request.form['2']
    bloodpressure = request.form['3']
    skinthickness = request.form['4']
    insulin = request.form['5']
    bmi = request.form['6']
    diabetespedigreefunction = request.form['7']
    age = request.form['8']
 

    #form a dataframe with the inpus and run the preprocessor as used in the training 
    row_df = pd.DataFrame([pd.Series([pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age])])
    row_df =  pd.DataFrame(std.transform(row_df))
	
    print(row_df)

    #predict the probability and return the probability of being a diabetic
    prediction=model.predict_proba(row_df)
    output='{0:.{1}f}'.format(prediction[0][1], 2)
    output_print = str(float(output)*100)+'%'
    if float(output)>0.5:
        return render_template('result.html',pred=f'You have a chance of having diabetes.\nProbability of you being a diabetic is {output_print}.\nEat clean and exercise regularly')
    else:
        return render_template('result.html',pred=f'Congratulations, you are safe.\n Probability of you being a diabetic is {output_print}')


if __name__ == '__main__':
    app.run(debug=True)
