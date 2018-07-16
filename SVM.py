import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import svm

iris = load_iris()
x  = iris.data[:,:2]
y = iris.target
print (x[0])
print (y[0])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=4)

svmClassifier = svm.SVC(kernel="linear")
svmClassifier.fit(X_train, y_train)

y_predict = svmClassifier.predict(X_test)
#plt.plot(y_test,y_predict,'g-',label="line")
print(len(X_test))
print(len(y_test))
print(len(y_predict))
print(X_test[:,0].shape)
print(y_test.shape)
print(y_predict.shape)

plt.scatter(X_test[:,0],y_predict,label="predicted", color="red", marker="x")
plt.show()
plt.scatter(X_test[:,0], y_test,label="original", color="blue", marker="o")
plt.show()

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print ('confusion metrix: ', confusion_matrix(y_test,y_predict))
print ('Accuracy Score :',accuracy_score(y_test, y_predict))
print ('Report : ')
print (classification_report(y_test, y_predict))
