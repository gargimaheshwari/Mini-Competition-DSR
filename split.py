import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

data = pd.read_csv("rossmann-store-sales/data.csv", index_col = 0)

tscv = TimeSeriesSplit(n_splits = 3)
idx1, idx2, idx3 = tscv.split(data)

train_idx, test_idx = idx3[0], idx3[1]

train_data = data.iloc[train_idx]
test_data = data.iloc[test_idx]

train_data.to_csv("rossmann-store-sales/train_data.csv")
test_data.to_csv("rossmann-store-sales/test_data.csv")