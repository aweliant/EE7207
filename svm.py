from sklearn import svm
from scipy.io import loadmat
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import random

data_train = loadmat('data_train.mat')['data_train']
data_test = loadmat('data_test.mat')['data_test']
label_train = loadmat('label_train.mat')['label_train'].squeeze()
# Feature Scaling
sc_X = StandardScaler()
data_train = sc_X.fit_transform(data_train)
data_test = sc_X.transform(data_test)


#Hyperparameter tuning
svc=svm.SVC(kernel = 'rbf', random_state = 0)
# C=[0.01,0.1,1,10,100]
# gamma=[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1]
C=np.arange(1,7,0.1)
gamma=np.arange(0.01,0.8,0.01)
param_grid = dict(C = C,gamma=gamma)
kfold = StratifiedKFold(n_splits=10, shuffle = True,random_state=5)
grid_search = GridSearchCV(svc,param_grid,scoring = 'f1',n_jobs = -1,cv = kfold)
grid_result = grid_search.fit(data_train, label_train)
print("Best: %s" % (grid_search.best_params_))


#evaluate SVM method by K-fold CV
index = [i for i in range(len(data_train))]
np.random.shuffle(index)
data_tr = data_train[index]
label_tr = label_train[index]
svc = svm.SVC(C=2.4, kernel='rbf', gamma=0.05, random_state=None)
# print(np.mean(cross_val_score(svc, data_tr, label_tr, cv=10, scoring='neg_mean_squared_error')))
print(np.mean(cross_val_score(svc, data_tr, label_tr, cv=10, scoring='accuracy')))
print(np.mean(cross_val_score(svc, data_tr, label_tr, cv=10, scoring='f1')))


#predict on testing dataset
classifier = svm.SVC(C=2.4, kernel='rbf', gamma=0.05, random_state=0)
classifier.fit(data_train, label_train)
y_pred = classifier.predict(data_test)
print(y_pred)
