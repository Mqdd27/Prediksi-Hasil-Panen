import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import Dense,  LSTM
from keras import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

data = pd.read_csv('Dataset/panen.csv')
data.shape
#X= 59 -> input
#y= 10 -> fitur/parameter

## Parameter/Fitur
X = data[['Bulan', 'curah hujan', 'Luas Panen (ha)','Luas Lahan']]
## Target
y = data['Produksi Padi (ton)']
   
"""
# Split
"""

# data['Tanggal'] = pd.to_datetime(data[['Tahun', 'Bulan']].assign(Day=1))

print (data)

"""
# Split
"""

test_split=round(len(data)*0.50)
data_training=data[:80]
data_testing=data[20:]
print(data_training.shape)
print(data_testing.shape)

data_training.shape

scaler = MinMaxScaler()
data_training_scaled = scaler.fit_transform(data_training)
data_testing_scaled=scaler.transform(data_testing)
data_training_scaled, data_testing_scaled

def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i, 2])
    return np.array(dataX),np.array(dataY)
X_train,y_train=createXY(data_training_scaled,6)
X_test,y_test=createXY(data_testing_scaled,6)

print("trainX Shape-- ",X_train.shape)
print("trainY Shape-- ",y_train.shape)

print("testX Shape-- ",X_test.shape)
print("testY Shape-- ",y_test.shape)

print("trainX[0]-- \n",X_train[0])
print("trainY[0]-- ",y_train[0])

#X_train.shape[1], X_test.shape[2], X_train.shape[0]
#X_train.shape[0], X_train.shape[1], X_train.shape[2]

from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn import metrics
from keras.callbacks import EarlyStopping


model = Sequential()
model.add(LSTM(4, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(8, return_sequences=False))
model.add(Dense(16))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))
model.compile(optimizer=AdamW(learning_rate=0.0001), loss='mean_squared_error')
# model.compile(loss=mean_squared_error, optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
model.fit(X_train, y_train, epochs=250, batch_size=8, validation_data=(X_test, y_test), callbacks=EarlyStopping(monitor='loss', patience=3))
# # model.fit(X_train, y_train, epochs=250, batch_size=8, validation_data=(X_test, y_test))
# model.summary()

prediction = model.predict(X_test)
print (prediction)
print (prediction.shape)

prediction_copies_array = np.repeat(prediction,5, axis=-1)
prediction_copies_array

# %%
prediction_copies_array.shape

# %%
label = scaler.inverse_transform(np.repeat(y_test.reshape(-1, 1), 5, axis=-1))[:, 2]

# %%
pred = scaler.inverse_transform(prediction_copies_array)[:, 2]

# %%
# pred=scaler.inverse_transform(np.reshape(prediction_copies_array,(len(prediction),4)))[:,2]

# %%
# original_copies_array = np.repeat(y_test,4, axis=-1)
# original=scaler.inverse_transform(np.reshape(original_copies_array,(len(y_test),4)))[:,2]

# %%
print("Pred Values-- " ,pred)
print("\nOriginal Values-- " ,label)

# %%
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error

mse = mean_squared_error(y_test, pred)

mae = mean_absolute_error(y_test, pred)

mape = mean_absolute_percentage_error(y_test,pred)

r2 = r2_score(y_test, pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Absolute Error (MAPE): {mape}")
print(f"r2 Score: {r2}")

# %%
# from scikeras.wrappers import KerasRegressor
# from sklearn.model_selection import GridSearchCV
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dropout, Dense

# # Define the build_model function with **kwargs to accept hyperparameters
# def build_model(**kwargs):
#     optimizer = kwargs.pop('optimizer', 'adam')  # Get the optimizer, default to 'adam'
#     model = Sequential()
#     model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
#     model.add(LSTM(50, return_sequences=False))
#     model.add(Dropout(0.2))
#     model.add(Dense(1))
#     model.compile(loss='mse', optimizer=optimizer)
#     return model

# # Create the KerasRegressor with build_fn
# grid_model = KerasRegressor(build_fn=build_model, verbose=1)

# # Define the parameter grid for GridSearchCV
# parameters = {
#     'batch_size': [16],
#     'epochs': [10],
#     'optimizer': ['adam']
# }

# # Create GridSearchCV
# grid_search = GridSearchCV(estimator=grid_model, param_grid=parameters, cv=2)
# grid_search = grid_search.fit(X_train, y_train)

# # Access the best model directly
# my_model = grid_search.best_estimator_

# # Make predictions
# prediction = my_model.predict(X_test)
# print("Prediction\n", prediction)
# print("\nPrediction Shape:", prediction.shape)

# %%
"""
## Plot Hasil Prediksi
"""
plt.plot(label, color = 'red', label = 'Real')
plt.plot(pred, color = 'blue', label = 'Predicted')
plt.title('Prediction')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
