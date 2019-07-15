import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from joblib import load

def metric(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])

def plot_feature_importances(rf, cols, model_dir):
    importances = pd.DataFrame()
    importances.loc[:, 'importances'] = rf.feature_importances_
    importances.loc[:, 'features'] = cols
    importances.sort_values('importances', inplace=True)
    
    f, a = plt.subplots()
    importances.plot(ax=a, kind='bar', x='features', y='importances')
    plt.gcf().subplots_adjust(bottom=0.3)
    f.savefig(os.path.join(model_dir, 'importances.png'))

def predictor(test_features, rf):
    return rf.predict(test_features)

test_data = pd.read_csv("rossmann-store-sales/test_data.csv", index_col = 0)
rf = load("xgb.joblib")

test_labels = np.array(test_data.loc[:, "Sales"])
test_features = test_data.drop("Sales", axis = 1)
model_dir = "C:\\Users\\Gargi\\Documents\\DataScience\\DSR\\Mini-competition\\"
features = test_features.columns

preds = predictor(test_features, rf)
print('RMSPE:', metric(preds, test_labels))
plot_feature_importances(rf, features, model_dir)

rf = load("xgb.joblib")
train_data = pd.read_csv("rossmann-store-sales/train_data.csv", index_col = 0)
train_labels = train_data.loc[:, "Sales"]
train_features = train_data.drop("Sales", axis = 1)
preds = predictor(train_features, rf)
print('RMSPE:', metric(preds, np.array(train_labels)))
