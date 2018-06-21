from sklearn import svm,datasets
iris=datasets.load_iris()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(iris['data'],iris['target'], random_state=0)
logreg=svm.SVC()
logreg.fit(X_train,y_train)
prediction=logreg.predict(X_test)
from sklearn.metrics import accuracy_score
y=accuracy_score(prediction,y_test)
print(y)