
from nimbusml.ensemble import LightGbmRegressor
from nimbusml.ensemble.booster import Gbdt
from nimbusml import Pipeline
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import random
import argparse


np.random.seed(26)

def fe(df: pd.DataFrame):
    df['c'] = pd.Series( np.random.choice(["A", "B", "C"], len(df))).astype('category')

    return df

def create_dataset():
    n_features = 5
    size=1000
    X = pd.DataFrame(np.random.normal(size=(size,n_features)), columns=[f'f{x}' for x in range(n_features)])
    y = pd.Series(np.random.standard_cauchy(size=size))

    return X, y

def nimbus_training(X, y):
    params = {
        "model": {
            "random_state": 26,
            "evaluation_metric": 'MeanAbsoluteError',
            "number_of_iterations": 100,
            "use_categorical_split": True
        },
        "booster": {
            "l1_regularization": 0.00000239,
            "l2_regularization": 0.0132,
            "feature_fraction": 0.98,
            "subsample_fraction": 0.99,
            "subsample_frequency": 5,
        }
    }
    model = Pipeline([ 
        LightGbmRegressor(booster=Gbdt(**params["booster"]), **params["model"])
    ])

    model.fit(X, y, verbose=100)

    return model

def nimbus_pred(model_path, test_set_path):
    X = pd.read_csv(test_set_path)
    X['c'] = X['c'].astype("category")
    p = Pipeline()
    p.load_model(model_path)
    pred = p.predict(X)
    print(pred)

parser = argparse.ArgumentParser()

parser.add_argument("-p", action="store_true")
if __name__ == "__main__":
    fp = lambda x: os.path.join(os.path.dirname(__file__), x)

    test_file_path = fp("dummy_test.csv")

    args = parser.parse_args()
    if args.p:
        nimbus_pred(fp('lgbm_nimbus.zip'), test_file_path)
    else:
        X, y = create_dataset()
        X = fe(X)
        X, X_test, y, y_test = train_test_split(X, y, train_size=0.99)

        model = nimbus_training(X, y)
        

        X_test.to_csv(test_file_path, index=False)

        model.save_model(fp("lgbm_nimbus.zip"))

        pd.DataFrame(model.predict(X_test)).to_csv(fp('dummy_test_scores_python.csv'), index=False)