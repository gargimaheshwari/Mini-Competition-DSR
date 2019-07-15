import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import xgboost

def metric(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])

def plot_feature_importances(rf, cols):
    importances = rf.get_fscore()
    importances = importances.items()
    df = pd.DataFrame(importances, columns = ['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    df.sort_values('fscore', inplace=True)
    
    
    f, a = plt.subplots()
    df.plot(ax = a, kind='bar', x='feature', y='fscore')
    plt.gcf().subplots_adjust(bottom=0.3)
    f.savefig('importances.png')

def predictor(test_features, rf):
    return rf.predict(xgboost.DMatrix(test_features))

#load data and model
test_data = pd.read_csv("rossmann-store-sales/new_test.csv", index_col = 0)
rf = load("xgb.joblib")

#Pre-processing
test_data = test_data[test_data["Open"] == 1]
test_data = test_data[test_data["Sales"] > 0]

test_data["Date"] = pd.to_datetime(test_data["Date"])
test_data["Month"] = test_data["Date"].dt.month
test_data["Day"] = test_data["Date"].dt.day
test_data["Year"] = test_data["Date"].dt.year

test_data.loc[:, "StateHoliday"] = test_data.loc[:, "StateHoliday"].map({0:0,'0':0, 'a':1,'b':2,'c':3})
#test_data.loc[:, "StoreType"] = test_data.loc[:, "StoreType"].map({'a':1,'b':2,'c':3, 'd':4})
#test_data.loc[:, "Assortment"] = test_data.loc[:, "Assortment"].map({'a':1,'b':2,'c':3})
#test_data.loc[:, "PromoInterval"] = test_data.loc[:,"PromoInterval"].map({"Jan,Apr,Jul,Oct":1, "Feb,May,Aug,Nov":2, "Mar,Jun,Sept,Dec":3})

#test_data.loc[:, "CompetitionDistance"].fillna(test_data.loc[:, "CompetitionDistance"].mean(), inplace = True)

features = ["Store", "DayOfWeek", "Sales", "Promo",
            "StateHoliday", "SchoolHoliday", 
            "Month", "Day", "Customers",
            "WeekOfYear", "Year"]

test_data = test_data.loc[:, features]

#Split columns
test_labels = np.array(test_data.loc[:, "Sales"])
test_features = test_data.drop("Sales", axis = 1)
features = test_features.columns

preds = predictor(test_features, rf)
print('RMSPE:', metric(preds, test_labels))
plot_feature_importances(rf, features)
