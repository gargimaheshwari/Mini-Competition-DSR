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

data.loc[:, "StateHoliday"] = data.loc[:, "StateHoliday"].map({0:0,'0':0, 'a':1,'b':2,'c':3})
data.loc[:, "StoreType"] = data.loc[:, "StoreType"].map({'a':1,'b':2,'c':3, 'd':4})
data.loc[:, "Assortment"] = data.loc[:, "Assortment"].map({'a':1,'b':2,'c':3})
#data.loc[:, "PromoInterval"] = data.loc[:,"PromoInterval"].map({"Jan,Apr,Jul,Oct":1, "Feb,May,Aug,Nov":2, "Mar,Jun,Sept,Dec":3})

data.loc[:, "CompetitionDistance"].fillna(data.loc[:, "CompetitionDistance"].mean(), inplace = True)

data.fillna(0, inplace=True)
data["WeekOfYear"] = data["Date"].dt.weekofyear
data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) + (data.Month - data.CompetitionOpenSinceMonth)
data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) + (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)
data.loc[data.Promo2SinceYear == 0, 'PromoOpen'] = 0
month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', \
             7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
data['monthStr'] = data.Month.map(month2str)
data.loc[data.PromoInterval == 0, 'PromoInterval'] = ''
data['IsPromoMonth'] = 0
for interval in data.PromoInterval.unique():
    if interval != '':
        for month in interval.split(','):
            data.loc[(data.monthStr == month) & (data.PromoInterval == interval), 'IsPromoMonth'] = 1


features = ["Store", "DayOfWeek", "Sales",
            "Open", "Promo",
            "StateHoliday", "SchoolHoliday",
            "StoreType", "Assortment",
            "CompetitionDistance", "Promo2", 
            "Month", "Day", "Customers",
            "WeekOfYear", "CompetitionOpen", "PromoOpen", "IsPromoMonth"]

data = data.loc[:, features]

data.to_csv("rossmann-store-sales/data.csv")