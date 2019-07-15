import pandas as pd

train = pd.read_csv("rossmann-store-sales/new_train.csv", index_col = 0, low_memory = False)
store = pd.read_csv("rossmann-store-sales/store.csv")
data = train.merge(store, on = "Store", how = "outer")

data = data[data["Open"] == 1]
data = data[data["Sales"] > 0]

data["Date"] = pd.to_datetime(data["Date"])
data["Month"] = data["Date"].dt.month
data["Day"] = data["Date"].dt.day
data["Year"] = data["Date"].dt.year
data["WeekOfYear"] = data["Date"].dt.weekofyear

data.loc[:, "StateHoliday"] = data.loc[:, "StateHoliday"].map({0:0,'0':0, 'a':1,'b':2,'c':3})
#data.loc[:, "StoreType"] = data.loc[:, "StoreType"].map({'a':1,'b':2,'c':3, 'd':4})
#data.loc[:, "Assortment"] = data.loc[:, "Assortment"].map({'a':1,'b':2,'c':3})
#data.loc[:, "PromoInterval"] = data.loc[:,"PromoInterval"].map({"Jan,Apr,Jul,Oct":1, "Feb,May,Aug,Nov":2, "Mar,Jun,Sept,Dec":3})

#data.loc[:, "CompetitionDistance"].fillna(data.loc[:, "CompetitionDistance"].mean(), inplace = True)

data.fillna(0, inplace=True)

features = ["Store", "DayOfWeek", "Sales", "Promo",
            "StateHoliday", "SchoolHoliday", 
            "Month", "Day", "Customers",
            "WeekOfYear", "Year"]

data = data.loc[:, features]

data.to_csv("rossmann-store-sales/data.csv")
