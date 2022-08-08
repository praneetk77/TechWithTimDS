import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pickle
import matplotlib.pyplot as pyplot
from matplotlib import style

data = pd.read_csv('student-mat.csv', sep=';')
req_columns = []
for x in data.columns:
    if data[x].dtype=='int64':
        req_columns.append(x)

data = data[req_columns]
predict = 'G3'

x = np.array(data.drop([predict],1)) #attributes
y = np.array(data[predict]) #labels
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)
best = 0

'''for _ in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)

    #skipping the training process and loading the model from the pickle file
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

    if acc>best:
        best = acc
        with open("studentmodel.pickle","wb") as f:
            pickle.dump(linear,f)'''

print("Best : ", best)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Constants : ", linear.coef_)
print("Intercept : ", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

style.use("ggplot")
p = "absences"
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final grade")
pyplot.show()