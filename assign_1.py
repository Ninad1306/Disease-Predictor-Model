import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

data = pd.read_csv('Training1.csv')

data['prognosis'] = data['prognosis'].str.strip()

Y = data['prognosis'].to_numpy()
X = data.drop(['prognosis'],axis=1)

transform = preprocessing.StandardScaler()
X = transform.fit_transform(X)

X_train,_, Y_train,_ = train_test_split( X, Y, test_size=0.95, random_state=2)

parameters ={"C":[0.01,0.1,1],'penalty':['l2'], 'solver':['lbfgs']}
lr=LogisticRegression()

logreg_cv = GridSearchCV(lr, parameters, cv=10)
logreg_cv.fit(X_train, Y_train)

@app.route('/predict', methods=['POST'])
def predict_disease():
    return jsonify({"disease": logreg_cv.predict(pd.DataFrame([request.get_json()]).values)[0]})


if __name__ == '__main__':
    app.run()