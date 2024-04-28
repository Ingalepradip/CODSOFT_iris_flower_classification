import pandas as pd
import pickle
from flask import Flask,render_template,request

application=Flask('__name__')
app=application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_point():
    if request.method=='GET':
        return render_template('index.html')
    else:
        sepal_length=float(request.form.get('sepal_length'))
        sepal_width=float(request.form.get('sepal_width'))
        petal_length=float(request.form.get('petal_length'))
        petal_width=float(request.form.get('petal_width'))

        new_df=pd.DataFrame([sepal_length,sepal_width,petal_length,petal_width]).T
        new_df.columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

        with open('NOTEBOOK/scaler.pkl','rb') as file1:
            pipe=pickle.load(file1)

        x_pre=pd.DataFrame(pipe.transform(new_df))
        x_pre.columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

        with open('NOTEBOOK/dc_best_predict.pkl','rb') as file2:
            model=pickle.load(file2)

        y_pred=model.predict(x_pre)

        with open('NOTEBOOK/encoder.pkl','rb') as file3:
            le=pickle.load(file3)

        y_pred_lb = le.inverse_transform(y_pred)[0]

        prob=model.predict_proba(x_pre).max()

        prediction=f'{y_pred_lb} with probability {prob}'

        return render_template('index.html',prediction=prediction)
    
if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)




