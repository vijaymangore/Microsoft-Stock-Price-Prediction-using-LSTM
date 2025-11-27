from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from datetime import datetime 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # SUPPRESS THE WARNING OF TENORFLOW
data=pd.read_csv("MicrosoftStock.csv")
# print(data.head())
# print(data.info())
# print(data.describe())
data['date'] = pd.to_datetime(data['date'])
# print(data.info())

#data visualization
# 1. Open and close value w.r.t. date
# plt.figure(figsize=(12,6))
# plt.plot(data["date"],data["open"],label="open",color="blue")
# plt.plot (data["date"],data["close"],label="close",color="red")
# plt.title("Open-close-over time")
# plt.legend()
# # plt.show()

# plt.figure(figsize=(12,6))
# plt.plot(data["date"],data["volume"],label="volume",color="pink")
# plt.title("stock volume w.r.t.time")
# plt.show()

numeric_data=data.select_dtypes(include=["int64","float64"])
plt.figure(figsize=(12,6))
sns.heatmap(numeric_data.corr(),annot=True,cmap="coolwarm")
plt.title("feature correlation heatmap")
# plt.show()

prediction=data.loc[
    (data['date']>datetime(2013,1,1)) &
    (data['date']<datetime(2018,1,1))
      ]      # filtered date

plt.figure(figsize=(12,6))
plt.plot(data['date'],data['close'],color='blue')
plt.xlabel(data["date"])
plt.ylabel(data["close"])
plt.title("closing price over time")
# plt.show()

# prepare for lstm model(sequential model)
stock_close=data['close']
dataset=stock_close.values # convertd into numpy
# print(dataset)

training_data_len=int(np.ceil(len(dataset)*0.95))

# preprocessing stages
scaler=StandardScaler()

scaled_data = scaler.fit_transform(dataset.reshape(-1, 1))



training_data=scaled_data[:training_data_len] #95% of all out data

X_train,y_train=[],[]

#Create a sliding windows for our stock (60 days)
for i in range(60,len(training_data)):
    X_train.append(training_data[i-60:i, 0])
    y_train.append(training_data[i,0])


X_train,y_train= np.array(X_train), np.array(y_train)

X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

model = keras.models.Sequential()

# First Layer
model.add(keras.layers.LSTM(64, return_sequences=True,input_shape=(X_train.shape[1],1)))

#second Layer
model.add(keras.layers.LSTM(64,return_sequences=False))

# Third layer
model.add(keras.layers.Dense(128,activation="relu"))

# Fourth layer
model.add(keras.layers.Dropout(0.50))

# Fifth layer
model.add(keras.layers.Dense(1))

model.summary()
model.compile(optimizer="adam",
              loss="mae",
              metrics=[keras.metrics.RootMeanSquaredError()])

training=model.fit(X_train,y_train,epochs=20,batch_size=32)


#prepare the test data

test_data = scaled_data[training_data_len-60:]
X_test,y_test=[],dataset[training_data_len: ]


for i in range(60,len(test_data)):
    X_test.append(test_data[i-60:i,0])


X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1 ))

# make a prediction
predictions= model.predict(X_test)
predictions=scaler.inverse_transform(predictions)

# Plotting data
train= data[:training_data_len]
test= data[training_data_len:]

test=test.copy()

test["Predictions"]=predictions


plt.figure(figsize=(12,6))
plt.plot(train['date'],train['close'],label="Train actual",color="blue")
plt.plot(test['date'],test['close'],label="Test actual",color="orange")
plt.plot(test['date'],test['Predictions'],label="predictions",color="red")
plt.title("Our stock predictions")
plt.xlabel("Date")
plt.ylabel("close price")
plt.legend()
plt.show()