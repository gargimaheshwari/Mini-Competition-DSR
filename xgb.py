import pandas as pd
import xgboost
from joblib import dump

def xgb_reg(train_data):
    
    train_labels = train_data.loc[:, "Sales"]
    train_features = train_data.drop("Sales", axis = 1)
    labels = xgboost.DMatrix(train_features, train_labels)
    
    params = {"eta": 0.3,
              "max_depth": 8,
              "seed": 123
              }
    
    xgb = xgboost.train(params, labels)
    return xgb

train_data = pd.read_csv("rossmann-store-sales/train_data.csv", index_col = 0)

xgb = xgb_reg(train_data)
dump(xgb, "xgb.joblib")
