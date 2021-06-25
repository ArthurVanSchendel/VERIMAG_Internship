# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import pickle

# For scaling, feature selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split 
from keras.callbacks import EarlyStopping
from tqdm.keras import TqdmCallback
import os

# For NN model
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.models import load_model
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
s_x = MinMaxScaler()
Xs = s_x.fit_transform(X)

s_y = MinMaxScaler()
ys = s_y.fit_transform(y)

X_FNN = []
y_FNN = []

window = 3

for i in range(window,len(df)):
    X_FNN.append(Xs[i])
    y_FNN.append(ys[i])

X_FNN, y_FNN = np.array(X_FNN), np.array(y_FNN)

print("\n shape of X_FNN: ", X_FNN.shape)
print("\n shape of y_FNN: ", y_FNN.shape)

# Split into train and test 
X_train, X_test, y_train, y_test = train_test_split(X_FNN, y_FNN, test_size=0.2, shuffle=False)



def build_model():
    model = Sequential()
    model.add(Dense(50, input_shape=[2], kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(40, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(3, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
    
model = build_model()

model.fit(X_train, y_train, validation_split=0.2, batch_size=5, epochs=40)

# Save model parameters
model_params = dict()
model_params['Xscale'] = s_x
model_params['yscale'] = s_y
model_params['window'] = window

pickle.dump(model_params, open('model_params.pkl', 'wb'))

# Predict using LSTM
yp_s = model.predict(X_test)


# Unscale data
Xtest_us = s_x.inverse_transform(X_test[:,:])
ytest_us = s_y.inverse_transform(y_test)

yp = s_y.inverse_transform(yp_s)

# Derive Tsp (setpoint) and T1 (sensor) from X data
sp = Xtest_us[:,0]
pv = Xtest_us[:,0] + Xtest_us[:,1]

mean_err_ffnn = 0 
max_err_ffnn = -9999
min_err_ffnn = 9999

for i in range(len(ytest_us)):
    error = abs(yp[i]-ytest_us[i])
    mean_err_ffnn += error

    if (error > max_err_ffnn):
        max_err_ffnn = error

    if (error < min_err_ffnn):
        min_err_ffnn = error

mean_err_ffnn = mean_err_ffnn / len(ytest_us)


print("\n the mean error for FFNN is = ", mean_err_ffnn)
print("\n the max error for FFNN is = ", max_err_ffnn)
print("\n the min error for FFNN is = ", min_err_ffnn)

# Plot SP, PID response, and LSTM response
plt.plot(sp,'k-',label='$SP$ $(^oC)$')
plt.plot(pv,'r-',label='$T_1$ $(^oC)$')
plt.plot(ytest_us,'b-',label='$Q_{PID}$ (%)')
plt.plot(yp,'g-',label='$Q_{FNN}$ (%)')
plt.legend(fontsize=12,loc='lower right')
plt.xlabel('Time',size=14)
plt.ylabel('Value',size=14)
plt.xticks(size=12)
plt.yticks(size=12);
plt.show()
