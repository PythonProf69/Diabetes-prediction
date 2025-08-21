import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

df=pd.read_csv('diabetes.csv')
# print(df.head())
# print(df.isnull().sum())
# print(df['Outcome'].value_counts())
x= df.drop('Outcome', axis=1)
y= df['Outcome']
scaler= StandardScaler()
x_scaled = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2,stratify=y, random_state=2)
model = svm.SVC(kernel='linear')
model.fit(x_train, y_train)
joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred)) 
print("Acc: ",accuracy_score(model.predict(x_train),y_train) * 100)


input_data=(6,148,72,35,0,33.6,0.627,50)
input_data_np=np.asarray(input_data)
input_data_reshaped=input_data_np.reshape(1,-1)
input_data_scaled=scaler.transform(input_data_reshaped)
prediction = model.predict(input_data_scaled)
print("Prediction for input data:", prediction[0])