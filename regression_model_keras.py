import pandas as pd
import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
from keras.models import Sequential
from keras.layers import Dense

concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
print(concrete_data.head())

print(concrete_data.shape)

#check for missing data in set

concrete_data.describe()

print(concrete_data.describe())

print(concrete_data.isnull().sum())

print('DATA PREPARATION: if fully populated data set it is ready for processing')

#split data into predictors and target
print('split data into predictors and target')

concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength

target = concrete_data['Strength'] # Strength column

print("predictors are <<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
print(predictors.head())
print("target is <<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
print(target.head())

print('normalize the data by substracting the mean and dividing by the standard deviation')

predictors_norm = (predictors - predictors.mean()) / predictors.std()
print('full dataset  gives us :<<<<<<<<<<<<<<<<<<<<<<<<<<<')
print(predictors_norm.head())

n_cols = predictors_norm.shape[1] 
# number of predictors
#we can use for building out network

print('MODELING 2 LAYER NETWORK<<<<<<<<<<<<<<<<<<<<<<<<<<<')
print('create a model that has two hidden layers, each of 50 hidden units.')
# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))  
    # Only the first layer needs the input_shape
    model.add(Dense(32, activation='relu'))  
    # Subsequent layers infer their input shape
    model.add(Dense(1))  
    # Final layer
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


print('TRAIN AND TEST THE NETWORK')
model = regression_model()
#Next, we will train and test the model at the same time using the fit method. We will leave out 30% of the data for validation and we will train the model for 100 epochs.

# fit the model
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)

#You can refer to this link for more information on the fit method: https://keras.io/models/sequential/

