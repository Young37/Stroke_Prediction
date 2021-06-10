#MLP
import pandas as pd 
csv_test = pd.read_csv('C:/Users/skvsn/Desktop/stroke/healthcare-dataset-stroke-data.csv',index_col='id')
csv_test = csv_test.dropna()
csv_test['gender'] = pd.Categorical(csv_test['gender'])
csv_test['gender'] = csv_test.gender.cat.codes
csv_test['ever_married'] = pd.Categorical(csv_test['ever_married'])
csv_test['ever_married'] = csv_test.ever_married.cat.codes
csv_test['work_type'] = pd.Categorical(csv_test['work_type'])
csv_test['work_type'] = csv_test.work_type.cat.codes
csv_test['Residence_type'] = pd.Categorical(csv_test['Residence_type'])
csv_test['Residence_type'] = csv_test.Residence_type.cat.codes
csv_test['smoking_status'] = pd.Categorical(csv_test['smoking_status'])
csv_test['smoking_status'] = csv_test.smoking_status.cat.codes
stroke = csv_test.pop('stroke')
csv_test.values
import tensorflow as tf
dataset = tf.data.Dataset.from_tensor_slices((csv_test.values, stroke.values))

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(csv_test.values, stroke.values, train_size = 0.6)
y_train = tf.keras.utils.to_categorical(y_train, 2)
y_test = tf.keras.utils.to_categorical(y_test, 2)

n_input = 10
n_hidden = 512
n_output = 2

mlp = Sequential()
mlp.add(Dense(units = n_hidden, activation = 'tanh',
              input_shape=(n_input,), kernel_initializer = 'random_uniform', bias_initializer = 'zeros'))
mlp.add(Dense(units = n_output, activation = 'tanh', kernel_initializer = 'random_uniform', bias_initializer = 'zeros'))

mlp.compile(loss = 'mean_squared_error', optimizer = Adam(learning_rate=0.001), metrics = ['accuracy'])
hist = mlp.fit(x_train, y_train, batch_size = 128, epochs = 30, validation_data = (x_test, y_test), verbose = 2)

res = mlp.evaluate(x_test, y_test, verbose = 0)
print("Accuracy is", res[1]*100)