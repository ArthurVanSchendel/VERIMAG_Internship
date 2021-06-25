import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import pickle

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
import os

from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.layers import LSTM, Dropout, Dense
import tensorflow as tf


# Load training data
df = pd.read_csv('PID_train_data.csv')

# Create new feature: setpoint error
df['err'] = df['Tsp'] - df['T1']

# Load possible features
X = df[['T1','Tsp','err']]
y = np.ravel(df[['Q1']])

# SelectKBest feature selection
bestfeatures = SelectKBest(score_func=f_regression, k='all')
fit = bestfeatures.fit(X,y)
#plt.bar(x=X.columns,height=fit.scores_);

X = df[['Tsp','err']].values
y = df[['Q1']].values


# Scale data
s_x2 = MinMaxScaler()
Xs2 = s_x2.fit_transform(X)

s_y2 = MinMaxScaler()
ys2 = s_y2.fit_transform(y)


# Keras LSTM model

# Hyperparameters for model
window = 10
layers = 2
batch_size = 100
drop = 0.1
units=50

def build_lstm():
    model2 = Sequential()
    # First layer specifies input_shape and returns sequences
    model2.add(LSTM(units=units, return_sequences=True, input_shape=(Xtrain.shape[1],Xtrain.shape[2])))
    model2.add(Dropout(rate=drop))
    # Middle layers return sequences
    for i in range(layers-2):
        model2.add(LSTM(units=units,return_sequences=True))
        model2.add(Dropout(rate=drop))
    # Last layer doesn't return anything
    #model2.add(Dropout(rate=drop))

    model2.add(LSTM(units=units))
    model2.add(Dropout(rate=drop))

    model2.add(Dense(1))
    model2.compile(optimizer='adam', loss='mean_squared_error')
    return model2


X_LSTM = []
y_LSTM = []

for i in range(window,len(df)):
    X_LSTM.append(Xs2[i-window:i])
    y_LSTM.append(ys2[i])

X_LSTM, y_LSTM = np.array(X_LSTM), np.array(y_LSTM)

Xtrain, Xtest, ytrain, ytest = train_test_split(X_LSTM, y_LSTM,test_size=0.2, shuffle=False)

model2 = build_lstm()

model2.fit(Xtrain, ytrain, validation_split=0.2, batch_size=5, epochs=30)

# Save model parameters
model_params2 = dict()
model_params2['Xscale'] = s_x2
model_params2['yscale'] = s_y2
model_params2['window'] = window

pickle.dump(model_params2, open('model_params2.pkl', 'wb'))

# Predict using LSTM
yp_s2 = model2.predict(Xtest)


# Unscale data
Xtest_us2 = s_x2.inverse_transform(Xtest[:,-1,:])
ytest_us2 = s_y2.inverse_transform(ytest)

yp2 = s_y2.inverse_transform(yp_s2)

sp2 = Xtest_us2[:,0]
pv2 = Xtest_us2[:,0] + Xtest_us2[:,1]

mean_err_lstm = 0 
max_err_lstm = -9999
min_err_lstm = 9999


for i in range(len(ytest_us2)):
    error2 = abs(yp2[i]-ytest_us2[i])
    mean_err_lstm += error2

    if (error2 > max_err_lstm):
        max_err_lstm = error2

    if (error2 < min_err_lstm):
        min_err_lstm = error2

mean_err_lstm = mean_err_lstm / len(ytest_us2)

print("\n the mean error for LSTM is = ", mean_err_lstm)
print("\n the max error for LSTM is = ", max_err_lstm)
print("\n the min error for LSTM is = ", min_err_lstm)

# Plot SP, PID response, and LSTM response
plt.plot(sp2,'k-',label='$SP$ $(^oC)$')
plt.plot(pv2,'r-',label='$T_1$ $(^oC)$')
plt.plot(ytest_us2,'b-',label='$Q_{PID}$ (%)')
plt.plot(yp2,'y-',label='$Q_{LSTM}$ (%)')
plt.legend(fontsize=12,loc='lower right')
plt.xlabel('Time',size=14)
plt.ylabel('Value',size=14)
plt.xticks(size=12)
plt.yticks(size=12);
plt.show()