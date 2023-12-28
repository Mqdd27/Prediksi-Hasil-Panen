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

df_yield = pd.read_csv('Dataset/bener.csv')
df_hujan = pd.read_csv('Dataset/curah hujan baru.csv')

## Data Curah Hujan
df_hujan.to_csv('udan.csv', index=False)
df_hujan_baru = pd.read_csv('udan.csv')
df_hujan_baru.info()

## Data Yield
df_yield.to_csv('panen.csv', index=False)
df_panen_baru = pd.read_csv('panen.csv')
df_panen_baru['Produksi Padi (ton)'] = pd.to_numeric(df_panen_baru['Produksi Padi (ton)'],errors='coerce')
df_panen_baru = df_panen_baru.loc[:, ~df_panen_baru.columns.str.contains('^Unnamed')]
df_panen_baru.info()

## Data Yield
df_yield.to_csv('panen.csv', index=False)
df_panen_baru = pd.read_csv('panen.csv')
df_panen_baru['Produksi Padi (ton)'] = pd.to_numeric(df_panen_baru['Produksi Padi (ton)'],errors='coerce')
df_panen_baru = df_panen_baru.loc[:, ~df_panen_baru.columns.str.contains('^Unnamed')]
data = pd.merge(df_panen_baru, df_hujan, on=['Bulan'])

data.shape
#X= 59 -> input
#y= 10 -> fitur/parameter
# iki opo wkwk
data
## Parameter/Fitur
X = data[['curah hujan (mm)', 'Luas Panen (ha)','Luas Lahan']]
## Target
y = data['Produksi Padi (ton)']

"""
# Split
"""
columns_to_drop = ['Bulan']
data = data.drop(columns_to_drop, axis=1)
data

data

# %%
"""
# Split
"""

# %%
test_split=round(len(data)*0.50)
data_training=data[:80]
data_testing=data[20:]
print(data_training.shape)
print(data_testing.shape)

# %%
data_training.shape

from sklearn.preprocessing import OneHotEncoder


# %%
def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i, 2])
    return np.array(dataX),np.array(dataY)
X_train,y_train=createXY(data_training_scaled,6)
X_test,y_test=createXY(data_testing_scaled,6)

# %%
print("trainX Shape-- ",X_train.shape)
print("trainY Shape-- ",y_train.shape)

# %%
print("testX Shape-- ",X_test.shape)
print("testY Shape-- ",y_test.shape)

# %%
print("trainX[0]-- \n",X_train[0])
print("trainY[0]-- ",y_train[0])

# %%
X_train.shape[1], X_test.shape[2], X_train.shape[0]

# %%
# from sklearn.model_selection import TimeSeriesSplit
# tss = TimeSeriesSplit(n_splits = 3)
# for train_index, test_index in tss.split(X):
#     X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# %%
# X_train = X_train.to_numpy()
# X_test = X_test.to_numpy()

# %%
X_train.shape[0], X_train.shape[1], X_train.shape[2]

# %%
"""
# RNN 
"""

# %%
# from scikeras.wrappers import KerasRegressor
# from sklearn.model_selection import GridSearchCV
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dropout, Dense
# from tensorflow.keras.optimizers import AdamW
# from tensorflow.keras.metrics import RootMeanSquaredError
# from sklearn import metrics
# from keras.callbacks import EarlyStopping


# model = Sequential()
# model.add(LSTM(4, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(LSTM(8, return_sequences=False))
# model.add(Dense(16))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='relu'))
# model.compile(optimizer=AdamW(learning_rate=0.0001), loss='mean_squared_error')
# # model.compile(loss=mean_squared_error, optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
# model.fit(X_train, y_train, epochs=250, batch_size=8, validation_data=(X_test, y_test), callbacks=EarlyStopping(monitor='loss', patience=3))
# # model.fit(X_train, y_train, epochs=250, batch_size=8, validation_data=(X_test, y_test))
# model.summary()

# %%
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam, Adadelta
from sklearn import metrics
from keras.callbacks import EarlyStopping

def build_model(optimizer=Adam(), batch_size=16, epochs=8):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(6, 4)))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=optimizer)
    return model

grid_model = KerasRegressor(build_fn=build_model, verbose=1)

parameters = {
    'batch_size': [16, 20],
    'epochs': [8, 10],
    'optimizer': [Adam(), Adadelta()]
}

grid_search = GridSearchCV(estimator=grid_model, param_grid=parameters, cv=2)

grid_search = grid_search.fit(X_train, y_train)


# %%
grid_search.best_params_

# %%
model=grid_search.best_estimator_.model

# %%
model.summary()

# %%
prediction=model.predict(X_test)
print("prediction\n", prediction)
print("\nPrediction Shape-",prediction.shape)

# %%
# train_loss = hist.history['loss']
# val_loss = hist.history['val_loss']

# epochs = range(1, len(train_loss) + 1)
# plt.plot(epochs, train_loss, label='Training Loss')
# plt.plot(epochs, val_loss, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# %%
prediction = model.predict(X_test)
prediction

# %%
prediction.shape

# %%
# scaler.inverse_transform(prediction)
prediction_copies_array = np.repeat(prediction,4, axis=-1)
prediction_copies_array

# %%
prediction_copies_array.shape

# %%
label = scaler.inverse_transform(np.repeat(y_test.reshape(-1, 1), 4, axis=-1))[:, 2]

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

# %%
# # Visualisasi data aktual dan prediksi dari model LSTM
# plt.figure(figsize=(10, 6))
# plt.plot(y_test, label='Aktual', marker='o')
# plt.plot(y_pred, label='Prediksi LSTM', linestyle='--', marker='x')
# plt.xlabel('Indeks Data')
# plt.ylabel('Nilai Target')
# plt.title('Perbandingan Data Aktual dan Prediksi dari Model LSTM')
# plt.legend()
# plt.grid(True)
# plt.show()

# %%
plt.plot(label, color = 'red', label = 'Real')
plt.plot(pred, color = 'blue', label = 'Predicted')
plt.title('Prediction')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# %%
