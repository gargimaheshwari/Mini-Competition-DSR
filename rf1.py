import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

def random_forest_fit(train_data):

    train_labels = train_data.loc[:, "Sales"]
    train_features = train_data.drop("Sales", axis = 1)

    rf = RandomForestRegressor(max_depth = 8, n_estimators = 100, random_state = 123)
    rf.fit(train_features, train_labels)
    return rf

train_data = pd.read_csv("rossmann-store-sales/train_data.csv", index_col = 0)

rf = random_forest_fit(train_data)
dump(rf, "rf1.joblib")