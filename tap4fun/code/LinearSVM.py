# coding:utf-8

import pandas as pd
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
try:
    from code.Script import *
except:
    from Script import *
import os

def read_data(files):
    train=pd.read_csv(files,index_col=0)
    columns=train.columns.tolist()
    dta=train[columns[:-1]]
    label=train[columns[-1]]
    return dta,label

if not os.path.exists("./model/svm"):
    os.makedirs('./model/svm')

X_train,y_train=read_data(train_file)
X_test,y_test=read_data(valid_file)

linear_svr=SVR(kernel='linear')   #线性核函数初始化的SVR
linear_svr.fit(X_train,y_train)
joblib.dump(linear_svr,"./model/svm/linear_svr.m")

linear_svr=joblib.load("./model/svm/linear_svr.m")
linear_svr_y_predict=linear_svr.predict(X_test)
print('R-squared value of linear SVR is',linear_svr.score(X_test,y_test))
print('The mean squared error of linear SVR is',mean_squared_error(y_test,linear_svr_y_predict))


poly_svr=SVR(kernel='poly')   #多项式核函数初始化的SVR
poly_svr.fit(X_train,y_train)
joblib.dump(poly_svr,"./model/svm/poly_svr.m")

print(' ')
poly_svr=joblib.load("./model/svm/poly_svr.m")
poly_svr_y_predict=poly_svr.predict(X_test)
print('R-squared value of Poly SVR is',poly_svr.score(X_test,y_test))
print('The mean squared error of Poly SVR is',mean_squared_error(y_test,poly_svr_y_predict))


rbf_svr=SVR(kernel='rbf')   #径向基核函数初始化的SVR
rbf_svr.fit(X_train,y_train)
joblib.dump(rbf_svr,"./model/svm/rbf_svr.m")

print(' ')
rbf_svr=joblib.load("./model/svm/rbf_svr.m")
rbf_svr_y_predict=rbf_svr.predict(X_test)
print('R-squared value of RBF SVR is',rbf_svr.score(X_test,y_test))
print('The mean squared error of RBF SVR is',mean_squared_error(y_test,rbf_svr_y_predict))


