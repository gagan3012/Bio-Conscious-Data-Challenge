import pandas as pd
import numpy as np
import re
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import pandas as pd
from numpy.random import seed

bg = pd.read_csv("https://raw.githubusercontent.com/gagan3012/Bio-Conscious-Data-Challenge/master/data/blood-glucose-data.csv")
da = pd.read_csv("https://raw.githubusercontent.com/gagan3012/Bio-Conscious-Data-Challenge/master/data/distance-activity-data.csv")
hr = pd.read_csv("https://raw.githubusercontent.com/gagan3012/Bio-Conscious-Data-Challenge/master/data/heart-rate-data.csv")

bg["point_timestamp"] = pd.to_datetime(bg.point_timestamp)
da["point_timestamp"] = pd.to_datetime(da.point_timestamp)
hr["point_timestamp"] = pd.to_datetime(hr.point_timestamp)

bg2 = bg.copy()
bg2.point_timestamp = pd.to_datetime(bg2['point_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').str[:17]+"00")
point_time = pd.date_range(start = bg2.point_timestamp.min(), end = bg2.point_timestamp.max(), freq = "1min")
point_time = pd.DataFrame({"point_timestamp" : point_time})
point_time["Check"] = 1

bg2 = point_time.merge(bg2, how = "outer", left_on = "point_timestamp", 
                       right_on = "point_timestamp").sort_values(by = ["point_timestamp"])

bg2 = bg2.drop(["Check", "timezone_offset"], axis = 1)
bg2 = bg2.rename(columns = {"point_value(mg/dL)": "point_value.mg.dL"})

bg2["point_value.mg.dL"] = bg2["point_value.mg.dL"].interpolate(method = "linear")
bg2["point_value.mg.dL"] = round(bg2["point_value.mg.dL"])
bg2 = bg2.groupby(["point_timestamp"]).mean()
bg2["point_timestamp"] = bg2.index
bg2.index = range(len(bg2))

hr2 = hr.copy()
hr2.point_timestamp = pd.to_datetime(hr2['point_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').str[:17]+"00")
point_time_hr = pd.date_range(start = hr2.point_timestamp.min(), end = hr2.point_timestamp.max(), freq = "1min")
point_time_hr = pd.DataFrame({"point_timestamp" : point_time_hr})
point_time_hr["Check"] = 1

hr2 = point_time_hr.merge(hr2, how = "outer", left_on = "point_timestamp", 
                       right_on = "point_timestamp").sort_values(by = ["point_timestamp"])

hr2 = hr2.drop(["Check", "timezone_offset"], axis = 1)

hr2["point_value"] = hr2["point_value"].interpolate(method = "linear")
hr2["point_value"] = round(hr2["point_value"])
hr2 = hr2.groupby(["point_timestamp"]).mean()
hr2["point_timestamp"] = hr2.index
hr2.index = range(len(hr2))

bg2 = bg2.merge(hr2, how = "left", left_on = "point_timestamp", right_on = "point_timestamp")
bg2.dropna(axis = 0, inplace = True)

da2 = da.copy()
da_iphone = da2[da2.device == "iPhone"]
da_fitbit = da2[da2.device == "FitbitWatch"]
da_iphone = da_iphone[da_iphone["point_value(kilometers)"] > 0]
da_fitbit = da_fitbit[da_fitbit["point_value(kilometers)"] > 0]

da_iphone.point_timestamp = pd.to_datetime(da_iphone['point_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').str[:17]+"00")
da_fitbit.point_timestamp = pd.to_datetime(da_fitbit['point_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').str[:17]+"00")

da_full = da_iphone.append(da_fitbit, ignore_index=True)
da_full = da_full.groupby(["point_timestamp"]).mean()
da_full["point_timestamp"] = da_full.index
da_full.index = range(len(da_full))
da_full = da_full.drop(["timezone_offset"], axis = 1)


point_time_da = pd.date_range(start = da_full.point_timestamp.min(), end = da_full.point_timestamp.max(), freq = "1min")
point_time_da = pd.DataFrame({"point_timestamp" : point_time_da})
point_time_da["Check"] = 1

da_full = point_time_da.merge(da_full, how = "outer", left_on = "point_timestamp", 
                       right_on = "point_timestamp").sort_values(by = ["point_timestamp"])

da_full = da_full.drop(["Check"], axis = 1)
da_full = da_full.rename(columns = {"point_value(kilometers)": "point_value.kilometers"})

da_full["point_value.kilometers"] = da_full["point_value.kilometers"].interpolate(method = "linear")

da_full = da_full.groupby(["point_timestamp"]).mean()
da_full["point_timestamp"] = da_full.index
da_full.index = range(len(da_full))

bg2 = bg2.merge(da_full, how = "left", left_on = "point_timestamp", right_on = "point_timestamp")

bg2 = bg2[0:len(bg2) - len(bg2) % 5]
bg2["grp"] = np.repeat(range(1,int((len(bg2) + 5) /5) ), 5)
bg2 = bg2.groupby(["grp"]).agg({"point_value.mg.dL": 'mean' , "point_value": 'mean',
                           "point_value.kilometers": 'mean', 'point_timestamp': 'min'})
bg2["grp"] = bg2.index
bg2.index = range(len(bg2))
bg2["point_value.mg.dL"] = round(bg2["point_value.mg.dL"])
bg2["point_value"] = round(bg2["point_value"])

bg2["future"] = pd.to_datetime(bg2['point_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').str[:17]+"00") + pd.Timedelta(minutes=5)

bg2 = bg2.merge(bg2[["point_timestamp", "point_value.mg.dL"]], how = "inner", left_on = "future", right_on = "point_timestamp")
bg2 = bg2.drop(columns = ["point_timestamp_y"], axis = 1)
bg2 = bg2.rename(columns = {"point_value.mg.dL_x" : "point_value.mg.dL", "point_value.mg.dL_y" : "Y"})

bg2 = bg2.dropna(how = "any")
bg2["maverage"] = bg2.loc[:, ["point_value.mg.dL"]].rolling(window=12, min_periods = 1).mean()
mavg = bg2["maverage"].tolist()
mavg.insert(0,np.nan)
mavg.pop(len(mavg)-1)
bg2["maverage"] = mavg
bg2.loc[0,"maverage"] = bg2.loc[1,"maverage"]
bg2["maverage"] = round(bg2["maverage"])

bg2 = bg2.rename(columns = {"point_timestamp_x" : "point_timestamp"})
bg2_train = bg2.loc[(bg2["point_timestamp"] <= "2017-05-29 23:59:00")]
bg2_Test = bg2.loc[(bg2["point_timestamp"] > "2017-06-01 23:59:00") & (bg2["point_timestamp"] <= "2017-07-07 23:59:00")]
bg2_actTest = bg[(bg["point_timestamp"] > "2017-06-01 23:59:00") & (bg["point_timestamp"] <= "2017-06-08 23:59:00")]
bg2_actTest["point_timestamp"] = pd.to_datetime(bg2_actTest['point_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').str[:17]+"00")

bg2_actTest = bg2_actTest.groupby("point_timestamp").mean()
bg2_actTest["point_timestamp"] = bg2_actTest.index
bg2_actTest.index = range(0,len(bg2_actTest))
bg2_actTest = bg2_actTest.drop(columns = ["timezone_offset"], axis = 1)

bg2_Test["maverage"] = np.nan
bg2_Test["maverage"] = bg2_Test.loc[:, ["point_value.mg.dL"]].rolling(window=12, min_periods = 1).mean()

mavg2 = bg2_Test["maverage"].tolist()
mavg2.insert(0,np.nan)
mavg2.pop(len(mavg2)-1)
bg2_Test["maverage"] = mavg2
bg2_Test = bg2_Test.reset_index()
bg2_Test.loc[0,"maverage"] = bg2_Test.loc[1,"maverage"]
bg2_Test["maverage"] = round(bg2_Test["maverage"])

bg2_train["speed"] = bg2_train["point_value.kilometers"]/(5/60)
bg2_Test["speed"] = bg2_Test["point_value.kilometers"]/(5/60)
bg2_train["day_night"] = np.where((bg2_train['point_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                                   .str[11:13].astype("int") >= 7) & 
                                  (bg2_train['point_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                                   .str[11:13].astype("int") < 19), 1,0)

bg2_Test["day_night"] = np.where((bg2_Test['point_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                                  .str[11:13].astype("int") >= 7)  & 
                                 (bg2_Test['point_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                                  .str[11:13].astype("int") < 19), 1,0)
bg2_actTest["point_timestamp"] = np.where((bg2_actTest['point_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').
                                           str[15:16].astype("int") == 2)
            | (bg2_actTest['point_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').str[15:16].astype("int") == 7),
            pd.to_datetime(bg2_actTest['point_timestamp'] - pd.Timedelta(minutes=1)),
                                              np.where((bg2_actTest['point_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').str[15:16].astype("int") == 0)
            | (bg2_actTest['point_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').str[15:16].astype("int") == 5), 
            pd.to_datetime(bg2_actTest['point_timestamp'] + pd.Timedelta(minutes=1)),bg2_actTest["point_timestamp"] + pd.Timedelta(minutes=0)))

bg2_train = bg2_train.drop(columns = ["grp"])
bg2_Test = bg2_Test.drop(columns = ["index","grp"])
a = pd.concat([bg2_train, bg2_Test])
scaled = a.copy()
scaled.loc[:,['point_value.mg.dL', 'point_value', 'point_value.kilometers',
        'Y', 'maverage', 'speed', 'day_night']] -= scaled.drop(columns = ["point_timestamp", "future"], axis=1).min()

scaled.loc[:,['point_value.mg.dL', 'point_value', 'point_value.kilometers',
        'Y', 'maverage', 'speed', 'day_night']] /= scaled.drop(columns = ["point_timestamp", "future"], axis=1).max()

scaled = scaled.drop(columns = ["point_timestamp", "future"])
scaled_train = scaled[0:len(bg2_train)]
scaled_test = scaled[(len(bg2_train)):len(scaled)]

scaled_test1 = scaled_test.copy()
scaled_test1["day_night"] = scaled_test1["day_night"].astype("int")
list_pred = []
seed(500)

model = Sequential()
model.add(Dense(1024, input_dim=6, activation='linear'))
model.add(Dense(512, activation='linear'))
model.add(Dense(64, activation='linear'))
model.add(Dense(1, activation='linear'))
sgd = optimizers.SGD(lr=0.1)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy', 'mse'] )

seed(300)
model.fit(scaled_train.loc[:, ['point_value.mg.dL', 'point_value', 'point_value.kilometers',
              'maverage', 'speed', 'day_night']], scaled_train.loc[:,"Y"], epochs=1000, batch_size=1000)

for i in range(1,13):
  pr_nn = model.predict(scaled_test1.drop(columns = ["Y"]))
  pr_nn1 = pr_nn * (max(a.Y) - min(a.Y)) + min(a.Y)
  pr_nn1 = pr_nn1.astype("int")
  list_pred.append(pr_nn1)
  scaled_test1["maverage"] = (scaled_test1["maverage"] + scaled_test1["point_value.mg.dL"]) / 2
  scaled_test1["point_value.mg.dL"] = pr_nn

nn_pred = pd.DataFrame.from_records(list_pred).transpose()
nn_pred = nn_pred.rename(columns = {0 : "Mins_5", 1 : "Mins_10", 2 : "Mins_15", 3 : "Mins_20", 4 : "Mins_25", 5 : "Mins_30", 6 : "Mins_35", 7 : "Mins_40",
                          8 : "Mins_45", 9 : "Mins_50", 10 : "Mins_55", 11 : "Mins_60"})

bg2_actTest["cc"] = 1
bg2_Test = bg2_Test.merge(bg2_actTest.loc[:,["point_timestamp","cc"]], how = "left",left_on = "point_timestamp", right_on = "point_timestamp" )

bg2_Test = bg2_Test.drop(columns = "Y")
bg2_Test["point_value.mg.dL"] = np.where((bg2_Test["cc"] == 1 ), bg2_Test["point_value.mg.dL"], np.nan)

bg2_Test = bg2_Test.merge(bg2_Test.loc[:,["point_timestamp", "point_value.mg.dL"]], how = "left", left_on = "future", 
                          right_on = "point_timestamp")
bg2_Test = bg2_Test.drop(columns = ['point_timestamp_y'])
bg2_Test = bg2_Test.rename(columns = {'point_value.mg.dL_x' : 'point_value.mg.dL', 'point_timestamp_x': 'point_timestamp', 
  'point_value.mg.dL_y' : 'Mins_5_actual'})

bg2_Test["future10"] = pd.to_datetime(bg2_Test['point_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').str[:17]+"00") + pd.Timedelta(minutes=10)
bg2_Test = bg2_Test.merge(bg2_Test.loc[:,["point_timestamp", "point_value.mg.dL"]], how = "left", left_on = "future10", 
                          right_on = "point_timestamp")
bg2_Test = bg2_Test.drop(columns = ['point_timestamp_y'])
bg2_Test = bg2_Test.rename(columns = {'point_value.mg.dL_x' : 'point_value.mg.dL', 'point_timestamp_x': 'point_timestamp', 
  'point_value.mg.dL_y' : 'Mins_10_actual'})

bg2_Test["future15"] = pd.to_datetime(bg2_Test['point_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').str[:17]+"00") + pd.Timedelta(minutes=15)
bg2_Test = bg2_Test.merge(bg2_Test.loc[:,["point_timestamp", "point_value.mg.dL"]], how = "left", left_on = "future15", 
                          right_on = "point_timestamp")
bg2_Test = bg2_Test.drop(columns = ['point_timestamp_y'])
bg2_Test = bg2_Test.rename(columns = {'point_value.mg.dL_x' : 'point_value.mg.dL', 'point_timestamp_x': 'point_timestamp', 
  'point_value.mg.dL_y' : 'Mins_15_actual'})

bg2_Test["future20"] = pd.to_datetime(bg2_Test['point_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').str[:17]+"00") + pd.Timedelta(minutes=20)
bg2_Test = bg2_Test.merge(bg2_Test.loc[:,["point_timestamp", "point_value.mg.dL"]], how = "left", left_on = "future20", 
                          right_on = "point_timestamp")
bg2_Test = bg2_Test.drop(columns = ['point_timestamp_y'])
bg2_Test = bg2_Test.rename(columns = {'point_value.mg.dL_x' : 'point_value.mg.dL', 'point_timestamp_x': 'point_timestamp', 
  'point_value.mg.dL_y' : 'Mins_20_actual'})

bg2_Test["future25"] = pd.to_datetime(bg2_Test['point_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').str[:17]+"00") + pd.Timedelta(minutes=25)
bg2_Test = bg2_Test.merge(bg2_Test.loc[:,["point_timestamp", "point_value.mg.dL"]], how = "left", left_on = "future25", 
                          right_on = "point_timestamp")
bg2_Test = bg2_Test.drop(columns = ['point_timestamp_y'])
bg2_Test = bg2_Test.rename(columns = {'point_value.mg.dL_x' : 'point_value.mg.dL', 'point_timestamp_x': 'point_timestamp', 
  'point_value.mg.dL_y' : 'Mins_25_actual'})

bg2_Test["future30"] = pd.to_datetime(bg2_Test['point_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').str[:17]+"00") + pd.Timedelta(minutes=30)
bg2_Test = bg2_Test.merge(bg2_Test.loc[:,["point_timestamp", "point_value.mg.dL"]], how = "left", left_on = "future30", 
                          right_on = "point_timestamp")
bg2_Test = bg2_Test.drop(columns = ['point_timestamp_y'])
bg2_Test = bg2_Test.rename(columns = {'point_value.mg.dL_x' : 'point_value.mg.dL', 'point_timestamp_x': 'point_timestamp', 
  'point_value.mg.dL_y' : 'Mins_30_actual'})

bg2_Test["future35"] = pd.to_datetime(bg2_Test['point_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').str[:17]+"00") + pd.Timedelta(minutes=35)
bg2_Test = bg2_Test.merge(bg2_Test.loc[:,["point_timestamp", "point_value.mg.dL"]], how = "left", left_on = "future35", 
                          right_on = "point_timestamp")
bg2_Test = bg2_Test.drop(columns = ['point_timestamp_y'])
bg2_Test = bg2_Test.rename(columns = {'point_value.mg.dL_x' : 'point_value.mg.dL', 'point_timestamp_x': 'point_timestamp', 
  'point_value.mg.dL_y' : 'Mins_35_actual'})

bg2_Test["future40"] = pd.to_datetime(bg2_Test['point_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').str[:17]+"00") + pd.Timedelta(minutes=40)
bg2_Test = bg2_Test.merge(bg2_Test.loc[:,["point_timestamp", "point_value.mg.dL"]], how = "left", left_on = "future40", 
                          right_on = "point_timestamp")
bg2_Test = bg2_Test.drop(columns = ['point_timestamp_y'])
bg2_Test = bg2_Test.rename(columns = {'point_value.mg.dL_x' : 'point_value.mg.dL', 'point_timestamp_x': 'point_timestamp', 
  'point_value.mg.dL_y' : 'Mins_40_actual'})

bg2_Test["future45"] = pd.to_datetime(bg2_Test['point_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').str[:17]+"00") + pd.Timedelta(minutes=45)
bg2_Test = bg2_Test.merge(bg2_Test.loc[:,["point_timestamp", "point_value.mg.dL"]], how = "left", left_on = "future45", 
                          right_on = "point_timestamp")
bg2_Test = bg2_Test.drop(columns = ['point_timestamp_y'])
bg2_Test = bg2_Test.rename(columns = {'point_value.mg.dL_x' : 'point_value.mg.dL', 'point_timestamp_x': 'point_timestamp', 
  'point_value.mg.dL_y' : 'Mins_45_actual'})

bg2_Test["future50"] = pd.to_datetime(bg2_Test['point_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').str[:17]+"00") + pd.Timedelta(minutes=50)
bg2_Test = bg2_Test.merge(bg2_Test.loc[:,["point_timestamp", "point_value.mg.dL"]], how = "left", left_on = "future50", 
                          right_on = "point_timestamp")
bg2_Test = bg2_Test.drop(columns = ['point_timestamp_y'])
bg2_Test = bg2_Test.rename(columns = {'point_value.mg.dL_x' : 'point_value.mg.dL', 'point_timestamp_x': 'point_timestamp', 
  'point_value.mg.dL_y' : 'Mins_50_actual'})

bg2_Test["future55"] = pd.to_datetime(bg2_Test['point_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').str[:17]+"00") + pd.Timedelta(minutes=55)
bg2_Test = bg2_Test.merge(bg2_Test.loc[:,["point_timestamp", "point_value.mg.dL"]], how = "left", left_on = "future55", 
                          right_on = "point_timestamp")
bg2_Test = bg2_Test.drop(columns = ['point_timestamp_y'])
bg2_Test = bg2_Test.rename(columns = {'point_value.mg.dL_x' : 'point_value.mg.dL', 'point_timestamp_x': 'point_timestamp', 
  'point_value.mg.dL_y' : 'Mins_55_actual'})

bg2_Test["future60"] = pd.to_datetime(bg2_Test['point_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').str[:17]+"00") + pd.Timedelta(minutes=60)
bg2_Test = bg2_Test.merge(bg2_Test.loc[:,["point_timestamp", "point_value.mg.dL"]], how = "left", left_on = "future60", 
                          right_on = "point_timestamp")
bg2_Test = bg2_Test.drop(columns = ['point_timestamp_y'])
bg2_Test = bg2_Test.rename(columns = {'point_value.mg.dL_x' : 'point_value.mg.dL', 'point_timestamp_x': 'point_timestamp', 
  'point_value.mg.dL_y' : 'Mins_60_actual'})

min_cols = [col for col in bg2_Test.columns if 'Mins_' in col]
act = bg2_Test.copy()
act = act.loc[:,min_cols]
nn_pred = nn_pred.astype("int")

diff_list = []
ard_list = []
for i in range(0,12):
  diff_list.append((act.iloc[:,i] - nn_pred.iloc[:,i])**2)
  ard_list.append( (abs(act.iloc[:,i] - nn_pred.iloc[:,i]) / nn_pred.iloc[:,i])*100 ) 
diff_tb = pd.DataFrame.from_records(diff_list).transpose()
ARD_tb = pd.DataFrame.from_records(ard_list).transpose()

rmse = (sum(diff_tb.sum()) / ((len(diff_tb.columns) * len(diff_tb)) - sum(len(diff_tb) - diff_tb.count())))**0.5
MARD = sum(ARD_tb.sum()) / ((len(ARD_tb.columns) * len(ARD_tb)) - sum(len(ARD_tb) - ARD_tb.count()))
print("RMSE:", round(rmse,4))
print("MARD:", round(MARD,4))