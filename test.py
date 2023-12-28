# %%
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

# %%
# df_ph = pd.read_csv('Dataset/Ph.csv')
# df_kelembapan = pd.read_csv('Dataset/kelembapan.csv')
df_yield = pd.read_csv('Dataset/Hasil Panen OKE sip.csv')
df_hujan = pd.read_csv('Dataset/curah hujan.csv')
# df_temp = pd.read_csv('Dataset/suhu tanah.csv')
# df_jarak = pd.read_csv('Dataset/jarak tanaman.csv')

# %%
## Data Curah Hujan
df_hujan.to_csv('udan.csv', index=False)
df_hujan_baru = pd.read_csv('udan.csv')
df_hujan_baru.info()

## Data Yield
df_yield.to_csv('panen.csv', index=False)
df_panen_baru = pd.read_csv('panen.csv')
df_panen_baru['Produksi Padi (ton/gkg)'] = pd.to_numeric(df_panen_baru['Produksi Padi (ton/gkg)'],errors='coerce')
df_panen_baru = df_panen_baru.loc[:, ~df_panen_baru.columns.str.contains('^Unnamed')]
df_panen_baru.info()
# df_panen_baru

# %%
## Data Yield
df_yield.to_csv('panen.csv', index=False)
df_panen_baru = pd.read_csv('panen.csv')
df_panen_baru['Produksi Padi (ton/gkg)'] = pd.to_numeric(df_panen_baru['Produksi Padi (ton/gkg)'],errors='coerce')
df_panen_baru = df_panen_baru.loc[:, ~df_panen_baru.columns.str.contains('^Unnamed')]
data = pd.merge(df_panen_baru, df_hujan, on=['Tahun', 'Bulan'])

# df_panen_baru.info()

# %%
data.shape
#X= 59 -> input
#y= 10 -> fitur/parameter

# %%
data

# %%
# month_mapping = {
#     'Januari': 1,
#     'Februari': 2,
#     'Maret': 3,
#     'April': 4,
#     'Mei': 5,
#     'Juni': 6,
#     'Juli': 7,
#     'Agustus': 8,
#     'September': 9,
#     'Oktober': 10,
#     'November': 11,
#     'Desember': 12
# }

# %%
## Parameter/Fitur
X = data[['curah hujan (mm)', 'Luas Panen (ha)','Luas Lahan']]
## Target
y = data['Produksi Padi (ton/gkg)']

# %%
"""
# Onehot encoding
"""

# %%
# from sklearn.preprocessing import OneHotEncoder
# df_onehot = pd.get_dummies(data, columns=['Tahun', 'Bulan'], prefix=['Tahun', 'Bulan'])
# data = df_onehot.loc[:, df_onehot.columns != 'Produksi Padi (ton/gkg)']
# data['Produksi Padi (ton/gkg)'] = df_onehot['Produksi Padi (ton/gkg)']

# %%
# data.shape

# %%
# data.dtypes

# %%
# data

# %%
"""
# Heat Map
"""

# %%
# corrmat = data.corr(method='pearson')
# cmap = sns.diverging_palette(260,-10,s=50, l=75, n=6, as_cmap=True)
# plt.subplots(figsize=(64,32))
# sns.heatmap(corrmat,cmap= cmap,annot=True, square=True)

# %%
"""
# Split
"""

# %%
month_mapping = {
    'Januari': 1,
    'Februari': 2,
    'Maret': 3,
    'April': 4,
    'Mei': 5,
    'Juni': 6,
    'Juli': 7,
    'Agustus': 8,
    'September': 9,
    'Oktober': 10,
    'November': 11,
    'Desember': 12
}

# %%
data['Bulan'] = data['Bulan'].map(month_mapping)

# %%
# data['Tanggal'] = pd.to_datetime(data[['Tahun', 'Bulan']].assign(Day=1))
data['Tanggal'] = pd.to_datetime(data['Tahun'].astype(str) + '-' + data['Bulan'].astype(str) + '-1', format='%Y-%m-%d')
columns_to_drop = ['Bulan', 'Tahun']
data = data.drop(columns_to_drop, axis=1)
data

# %%
data.set_index('Tanggal', inplace=True)
data.sort_index(inplace=True)
data

# %%
"""
# Split
"""

# %%
test_split=round(len(data)*0.20)
data_training=data[:30]
data_testing=data[30:]
print(data_training.shape)
print(data_testing.shape)

# %%
data_training.shape

# %%
scaler = MinMaxScaler()
data_training_scaled = scaler.fit_transform(data_training)
data_testing_scaled=scaler.transform(data_testing)
data_training_scaled, data_testing_scaled

# %%
def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i, 2])
    return np.array(dataX),np.array(dataY)
X_train,y_train=createXY(data_training_scaled,2)
X_test,y_test=createXY(data_testing_scaled,2)

# %%


# %%


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
# X_train.shape[0]

# %%
"""
# RNN 
"""

# %%
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from keras.callbacks import EarlyStopping


model = Sequential()
model.add(LSTM(128, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
# model.compile(loss=mean_squared_error, optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
model.fit(X_train, y_train, epochs=250, batch_size=8, callbacks=EarlyStopping(monitor='loss'))
model.summary()

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
