import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB  # Gaussian naive Bayes classifier

iris = load_iris()
x  = iris.data
y = iris.target
print (x[0])
print (y[0])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)

pClassifier = GaussianNB()
pClassifier.fit(X_train, y_train)

y_predict = pClassifier.predict(X_test)
#plt.plot(y_test,y_predict,'g-',label="line")
print(len(X_test))
print(len(y_test))
print(len(y_predict))
print(X_test[:,0].shape)
print(y_test.shape)
print(y_predict.shape)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print ('confusion metrix: ', confusion_matrix(y_test,y_predict))
print ('Accuracy Score :',accuracy_score(y_test, y_predict))
print ('Report : ')
print (classification_report(y_test, y_predict))

plt.scatter(X_test[:,0],y_predict,label="predicted", color="red", marker="x")
plt.scatter(X_test[:,0], y_test,label="original", color="blue", marker="o")
plt.show()
