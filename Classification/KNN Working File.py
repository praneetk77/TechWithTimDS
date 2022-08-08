import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
cls = le.fit_transform(list(data["class"]))
safety = le.fit_transform(list(data["safety"]))

predict = "class"

x = list(zip(buying,maint,door,persons,lug_boot,safety))
y = list(cls)
print(y)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)

k = 9 #highest for 9, tested from 5 to 15

model = KNeighborsClassifier(n_neighbors=k)
model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
print(acc)

names = ['unacc','acc','good','vgood']
predicted = model.predict(x_test)

for i in range(len(x_test)):
    print("Predicted : ", names[predicted[i]], " Parameters : ", x_test[i]," Actual value : ", names[y_test[i]])
    n = model.kneighbors([x_test[i]],k,True)
    print("Nearest neighbours : ", n)

