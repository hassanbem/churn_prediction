from flask import Flask,request,jsonify
import joblib
import pandas as pd


app=Flask(__name__)

model=joblib.load("churn_model.pkl")
columns=joblib.load("model_columns.pkl")
@app.route("/predict",methods=['POST'])
def predict():
    data=request.get_json()
    df=pd.DataFrame.from_dict(data)
    df=df.reindex(columns=columns,fill_value=0)
    y=model.predict(df)
    return jsonify({"prediction":y.tolist()})
if __name__=="__main__":
    app.run(debug=True)

