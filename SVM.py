#SVM
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

from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
x_train, x_test, y_train, y_test = train_test_split(csv_test.values, stroke.values, train_size = 0.6)

s = svm.SVC(gamma=0.001)
s.fit(x_train, y_train) # 학습(모델)

res = s.predict(x_test)

conf = np.zeros((2,2))
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1
print(conf)

correct = 0
for i in range(2):
    correct += conf[i][i]
accuracy = correct/len(res)
print("Accuracy is", accuracy*100,"%")



