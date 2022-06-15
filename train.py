import os
import yaml
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

params = yaml.safe_load(open("configs.yaml"))["train"]
seed = params["seed"]
n_est = params["n_est"]

def fit_model(data):
    rf = RandomForestClassifier(n_estimators = n_est, random_state = seed)
    rf.fit(data.drop(["PassengerId", "Survived"], axis = 1), data["Survived"])

    # save
    joblib.dump(rf, "model.pkl") 
    return

def make_predictions(model_name, data):
    # load
    rf = joblib.load(model_name)
    
    preds = rf.predict(data.drop(["PassengerId", "Survived"], axis = 1))
    json_obj = {}
    json_obj["Accuracy"] = np.mean(preds == data["Survived"])
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(json_obj, f, ensure_ascii=False, indent=4)
    return
 
features_path = os.path.join(os.getcwd(), "features")
train_df = pd.read_csv(os.path.join(features_path, "train_features.csv"), index_col=[0])
fit_model(train_df)
make_predictions("model.pkl", train_df)