import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, jsonify
from sklearn.svm import SVC
import numpy as np
from flask_cors import CORS,cross_origin
import json

app = Flask(__name__)
CORS(app)
cors = CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})


data = pd.read_csv('Training1.csv')

data['prognosis'] = data['prognosis'].str.strip()

Y = data[['prognosis']]
X = data.drop(['prognosis'],axis=1)

# transform = preprocessing.StandardScaler()
# X = transform.fit_transform(X)

X_train,_, Y_train,_ = train_test_split( X, Y, test_size=0.2, random_state=101)

# parameters ={"C":[0.01,0.1,1],'penalty':['l2'], 'solver':['lbfgs']}
# lr=LogisticRegression()

# logreg_cv = GridSearchCV(lr, parameters, cv=10)
# logreg_cv.fit(X_train, Y_train)

svc_model = SVC()
svc_model.fit(X_train.values,Y_train)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict_disease():
    # return jsonify({"disease": logreg_cv.predict(pd.DataFrame([request.get_json()]).values)[0]})
    # print(svc_model.decision_function(pd.DataFrame([request.get_json()]).values))
    disease_arr = request.get_json()['symptoms']
    with open("symptoms.json", "r") as file:
        data = json.load(file)
        for i in disease_arr:
            data[i]=1
    # return jsonify({"disease": svc_model.predict(pd.DataFrame([request.get_json()]).values)[0], "score": '1'})
        return jsonify({"disease": svc_model.predict(pd.DataFrame([data]).values)[0], "score": '1'})
if __name__ == '__main__':
    app.run(debug=True)