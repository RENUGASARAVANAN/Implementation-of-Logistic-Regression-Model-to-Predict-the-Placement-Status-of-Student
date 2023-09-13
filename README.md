# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:S.RENUGA
RegisterNumber: 212222230118

import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#removes the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1


x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")#A Library for Large Linear Classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:
![Screenshot (44)](https://github.com/RENUGASARAVANAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119292258/f725e85a-732e-4a4d-a178-e1a699820c9f)

![Screenshot (45)](https://github.com/RENUGASARAVANAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119292258/39c9dc01-c66a-46e5-9333-d230cdfa5ef5)

![Screenshot 2023-09-13 084858](https://github.com/RENUGASARAVANAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119292258/5c1bf0c1-4caa-43c4-9ff4-f31f25b3122f)

![Screenshot 2023-09-13 085046](https://github.com/RENUGASARAVANAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119292258/871cd4b3-dd43-41a4-8be2-56320097cce3)


![Screenshot 2023-09-13 085208](https://github.com/RENUGASARAVANAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119292258/abb468c7-e279-475d-b4c8-44fbb4ff6206)


![Screenshot 2023-09-13 085252](https://github.com/RENUGASARAVANAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119292258/9cce55e9-3a62-47a6-aefb-19ddeb173b36)


![Screenshot 2023-09-13 085338](https://github.com/RENUGASARAVANAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119292258/f756b5d9-89dd-4224-b8bb-c1122399986f)


![Screenshot 2023-09-13 085419](https://github.com/RENUGASARAVANAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119292258/07cceb4a-193a-43dd-8fba-d8c862f1352a)

![Screenshot 2023-09-13 085501](https://github.com/RENUGASARAVANAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119292258/5908fdc5-72fe-4c8b-a8c7-a4ec979a3348)

![Screenshot 2023-09-13 085538](https://github.com/RENUGASARAVANAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119292258/b363f111-c807-49be-ad99-65cb5797aeee)

![Screenshot 2023-09-13 085642](https://github.com/RENUGASARAVANAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119292258/d42f9ca3-7ebc-493e-934e-52367055d00d)


![Screenshot 2023-09-13 085740](https://github.com/RENUGASARAVANAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119292258/3504f8cb-b14f-4acd-b0dc-86ba5ae71d5f)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
