from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,GridSearchCV
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report


cancer=load_breast_cancer()
cancer.keys()
# print(cancer['DESCR'])
df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
# print(cancer['target_names'])

X=df
y=cancer['target']
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.33)
model=SVC()
model.fit(x_train,y_train)
pred=model.predict(x_test)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

grid={
    'C':[0.1,1,10,100,1000],
    'gamma':[1,0.1,0.01,0.001,0.0001]
}
model=GridSearchCV(SVC(),grid,verbose=3)
model.fit(x_train,y_train)
model.best_estimator_
model.best_params_
pred=model.predict(x_test)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

