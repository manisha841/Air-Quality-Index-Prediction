from typing import final
import pandas as pd
import numpy as np
import os

final_df=pd.read_csv("prepared_final_data.csv")

print(final_df)
values=final_df["pollution"].values
print(values)
print(final_df.columns)
"""# Normalized the data"""

from sklearn.preprocessing import MinMaxScaler

# values = final_df.values
print(values)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_dataset = scaler.fit_transform(values.reshape(-1,1))

scaled_dataset

# Creating a window for previous data
def to_supervised(window_size,train):
  X = []
  Y = []
  for i in range(window_size, len(train)):
    X.append(train[i-window_size:i,:])
    Y.append(train[i,0:1])
    
  return np.array(X), np.array(Y)

feature,label = to_supervised(window_size=5, train=scaled_dataset)

feature.shape, label.shape

n_train = 24*365
X_train, X_test = feature[n_train:,] , feature[:n_train,]
print('X_train' ,X_train.shape)
print('X_test' ,X_test.shape)

Y_train, Y_test = label[n_train:,] , label[:n_train,]
print('Y_train' ,Y_train.shape)
print('Y_test' ,Y_test.shape)


import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout,LSTM

model = Sequential()

model.add(LSTM(units = 50, return_sequences = True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

from keras.callbacks import EarlyStopping

es_callback = EarlyStopping(monitor='val_loss', patience=3,min_delta=0.01)
path = 'air_pollution_forecasting_model'
  
isdir = os.path.isdir(path)
print(isdir)
if isdir:
    reconstructed_model = keras.models.load_model("air_pollution_forecasting_model")
    model = reconstructed_model
else:
    model.fit(X_train, Y_train, validation_split = 0.1, epochs = 10, batch_size = 32, callbacks=[es_callback])
    model.save("air_pollution_forecasting_model")

Y_pred = np.round(model.predict(X_test),2)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y_test, Y_pred)

rmse = np.sqrt(mse)
print(rmse)

# Scaling back to the original scale
d = scaled_dataset[:8760,:]
print('dummy',d.shape)
print('Y_pred',Y_pred.shape)
Y_predicted = np.concatenate((Y_pred,d[:8760,1:]), axis =1)
print('concat y_pred',Y_pred.shape)
Y_tested = np.concatenate((Y_test, d[:8760,1:]), axis = 1)
print('concat Y_test', Y_test.shape)

Y_predicted = scaler.inverse_transform(Y_predicted)
Y_tested = scaler.inverse_transform(Y_tested)
Y_predicted = Y_predicted[:,0:1]
Y_tested = Y_tested[:,0:1]
print('Y_tested', Y_tested.shape)
print('Y_predicted', Y_predicted.shape)

import matplotlib.pyplot as plt

plt.plot(Y_predicted[:100,:], color= 'green')
plt.plot(Y_tested[:100,:] , color = 'red')
plt.title("Air Pollution Prediction (Multivariate)")
plt.xlabel("Date")
plt.ylabel("Pollution level")
plt.savefig("results.png")

import pickle
pickle.dump(scaler, open('min_max_scaler.pkl','wb'))
