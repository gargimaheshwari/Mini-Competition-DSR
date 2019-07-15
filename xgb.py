import pandas as pd
from xgboost.sklearn import XGBRegressor
from joblib import dump

def xgb_reg(train_data):
    
    train_labels = train_data.loc[:, "Sales"]
    train_features = train_data.drop("Sales", axis = 1)

    xgb = XGBRegressor(eta = 0.3, max_depth = 8, 
                       #colsample_bytree = 0.7, 
                       #sub_sample = 0.9, 
                       seed = 123)
    xgb.fit(train_features, train_labels)
    return xgb

train_data = pd.read_csv("rossmann-store-sales/train_data.csv", index_col = 0)

xgb = xgb_reg(train_data)
dump(xgb, "xgb.joblib")