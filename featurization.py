import os
import pandas as pd
from category_encoders import target_encoder, cat_boost
from sklearn import preprocessing
import numpy as np
from sklearn.impute import SimpleImputer

data_path = os.path.join(os.getcwd(), "data")
train = pd.read_csv(os.path.join(data_path, "train.csv"), index_col=[0])
test = pd.read_csv(os.path.join(data_path, "test.csv"), index_col=[0])

# Clean variables
def dummify_cols(df):
    embarked_df = pd.get_dummies(df[["PassengerId","Embarked"]])
    df = df.merge(embarked_df, on = "PassengerId", how = "inner")
    df = df.drop(["Embarked"], axis = 1)
    return df

def processing_cols(train, test):
    le = preprocessing.LabelEncoder()
    train["Sex"] = le.fit_transform(train["Sex"])
    test["Sex"] = le.transform(test["Sex"])

    te = target_encoder.TargetEncoder()
    train["Cabin"] = te.fit_transform(train["Cabin"], train["Survived"])
    test["Cabin"] = te.transform(test["Cabin"])

    train["Prefix"] = train["Name"].apply(lambda x: x.split(", ")[1].split()[0][:-1])
    test["Prefix"] = test["Name"].apply(lambda x: x.split(", ")[1].split()[0][:-1])
    ce = cat_boost.CatBoostEncoder()
    train["Prefix"] = ce.fit_transform(train["Prefix"], train["Survived"])
    test["Prefix"] = ce.transform(test["Prefix"])
    
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    train["Age"] = imp.fit_transform(np.array(train["Age"]).reshape(-1,1))
    test["Age"] = imp.transform(np.array(test["Age"]).reshape(-1,1))
    
    train = train.drop(["Name"], axis = 1)
    test = test.drop(["Name"], axis = 1)
    
    ne = preprocessing.StandardScaler()
    train["Fare"] = ne.fit_transform(np.array(train["Fare"]).reshape(-1,1))
    test["Fare"] = ne.transform(np.array(test["Fare"]).reshape(-1,1))
    train["Age"] = ne.fit_transform(np.array(train["Age"]).reshape(-1,1))
    test["Age"] = ne.fit_transform(np.array(test["Age"]).reshape(-1,1))
    
    return train, test


train = dummify_cols(train)
test = dummify_cols(test)
train, test = processing_cols(train, test)

os.makedirs(os.path.join(os.getcwd(), "features"), exist_ok=True)
features_path = os.path.join(os.getcwd(), "features")
train.to_csv(os.path.join(features_path, "train_features.csv"))
test.to_csv(os.path.join(features_path, "test_features.csv"))