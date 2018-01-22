

import requests
import bs4
import time
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, FLOAT, INTEGER,create_engine
from sqlalchemy.orm import sessionmaker
from apscheduler.schedulers.blocking import BlockingScheduler  # Time scheduler
import time
import numpy as np
from sklearn.externals import joblib
import neurolab as nl

from scipy.optimize import minimize

from sqlalchemy import create_engine,and_
from sqlalchemy import Column,DATETIME,FLOAT,String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sklearn.svm import SVR

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from math import sqrt
from matplotlib import pyplot
from numpy import array
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
Base = declarative_base()
class one_minute_history_data(Base):
    # one minute data
    __tablename__ = 'one_minute_data'
    TIME_STAMP = Column(INTEGER, primary_key=True)
    AC_PD = Column(FLOAT)
    AC_PD_f = Column(FLOAT)
    AC_QD = Column(FLOAT)
    NAC_PD = Column(FLOAT)
    NAC_QD = Column(FLOAT)
    DC_PD = Column(FLOAT)
    NDC_PD = Column(FLOAT)
    PV_PG = Column(FLOAT)
    PV_PG_f=Column(FLOAT)
    WP_PG = Column(FLOAT)



def read_data_load(train_start,train_end,test_start,test_end):
	ems_engine = create_engine('mysql://root:3333@localhost/emsdb')
	DBsession = sessionmaker(bind=ems_engine)
	ems_session=DBsession()

	train_data = ems_session.query(ForecastDBData.Demand).\
		filter(and_(ForecastDBData.date>=train_start,ForecastDBData.date<=train_end)).all()
	print('..................................................train data')
	print(train_data)
	train_len = len(train_data)

	test_data = ems_session.query(ForecastDBData.Demand). \
		filter(and_(ForecastDBData.date >= test_start, ForecastDBData.date <= test_end)).all()
	test_len = len(test_data)
	print('..................................................array')
	a = []
	for i in range(0,train_len):
		a.append(train_data[i][0])
#	b = [i[0] for i in train_data]
	print(a)
	for i in range(0,test_len):
		a.append(test_data[i][0])
	eload = a
	#data = np.vstack((np.asarray(train_data),np.asarray(test_data)))
	#eload = data.reshape(-1,1)
#	load1=data.shape[1]

	#print('..................................................data')
	#print(load1)

	ems_session.close()
	return eload,train_len,test_len

# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)-1):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


def prepare_data(raw_values, n_test, n_lag, n_seq):
    # extract raw values
       # transform data to be stationary
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), 1)
    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(diff_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
    # split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, train, test

# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # design network
    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2])
                   , stateful=True))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit network
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        model.reset_states()
    model.save('my_EMCmodel12.h5')
    return model

# make one forecast with an LSTM,
def forecast_lstm(model,X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, 1, len(X))
    # make forecast
  #  model = load_model('my_EMCmodel.h5')
    forecast = model.predict(X, batch_size=n_batch)
    # convert to array
    # print('forecast')
    # print('forecast')

    return [x for x in forecast[0, :]]


# evaluate the persistence model
def make_forecasts(model,n_batch, train, test, n_lag, n_seq):
    forecasts = list()
    for i in range(len(test)):
        X, y = test[i, 0:n_lag], test[i, n_lag:]
        # make forecast
        forecast = forecast_lstm(model, X, n_batch)
        # store the forecast
        print(i)
        print(forecast)
        forecasts.append(forecast)
    return forecasts


# invert differenced forecast
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i - 1])
    return inverted

# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test,n_seq):
    inverted = list()
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # invert differencing
        index = len(series) - n_test -n_seq + i
        print('index')
        print(index)
        last_ob = series[index]
        inv_diff = inverse_difference(last_ob, inv_scale)
        # store
        inverted.append(inv_diff)
    return inverted

# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    for i in range(n_seq):
        actual = [row[i] for row in test]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = sqrt(mean_squared_error(actual, predicted))
        print('t+%d RMSE: %f' % ((i + 1), rmse))


# store forecast

train_start = 1512288000
train_end = 1512288000+30*60*1200
test_start = 1512288000+30*60*1201
test_end = 1512288000+30*60*1320

n_lag = 120
n_seq = 48
n_epochs = 1000
n_batch = 1
n_neurons = 1


power, train_len,test_len = read_data_load(train_start,train_end,test_start,test_end)

n_test = test_len - n_seq

scaler, train, test = prepare_data(power, n_test, n_lag, n_seq)
print(train)

model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
# make forecasts
forecasts = make_forecasts(model,n_batch, train, test, n_lag, n_seq)
# inverse transform forecasts and test
forecasts = inverse_transform(power, forecasts, scaler, n_test, n_seq)
actual = [row[n_lag:] for row in test]
actual = inverse_transform(power, actual, scaler, n_test-1, n_seq)
# evaluate forecasts
evaluate_forecasts(actual, forecasts, n_lag, n_seq)
# plot forecasts

ems_engine = create_engine('mysql://root:3333@localhost/emsdb')
DBsession = sessionmaker(bind=ems_engine)
ems_session = DBsession()
selected_dates = ems_session.query(ForecastDBData.date). \
 	filter(and_(ForecastDBData.date >= test_start, ForecastDBData.date <= test_end)).all()

for i in range(0, len(forecasts)):
    row1 = ems_session.query(ForecastDBData).filter_by(date=selected_dates[i][0]).first()
    row1.Demand_f = forecasts[i][0]
    row2 = ems_session.query(ForecastDBData).filter_by(date=selected_dates[i+1][0]).first()
    row2.Demand_f2 = forecasts[i][1]
    print(i)
    print(row1.Demand_f)
    print('>>>')
ems_session.commit()
ems_session.close()


