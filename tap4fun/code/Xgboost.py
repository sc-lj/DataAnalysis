# coding:utf-8
try:
    from code.Script import *
    from code.MLearn import *
except:
    from Script import *
    from MLearn import *

import xgboost as xgb
from xgboost import XGBRegressor,XGBClassifier
from sklearn import metrics



def modelfit(alg,X_train,Y_train,X_valid,Y_valid,useTrainCV=True,cv_folds=5,early_stopping_rounds=50):
    alg.fit(X_train,Y_train,eval_metric='rmse',eval_set=[(X_valid,Y_valid)])
    train_predicts=alg.predict(X_valid)
    print("这个模型的mse: ",metrics.mean_squared_error(Y_valid,train_predicts))
    print("这个模型的r2_score: ",metrics.r2_score(Y_valid,train_predicts))

"""训练Xgboost 回归模型"""
def xgboostF():
    traindata, traintarget=getFeature(train_file)
    validdata,validtarget=getFeature(valid_file)
    # 加载array格式的数据到xgb中。
    dtrain = xgb.DMatrix(traindata, label=traintarget)
    dvalid=xgb.DMatrix(validdata,label=validtarget)

    # 参数
    param={
        "objective":"reg:logistic",   # 回归问题
        "eval_metric":"rmse",         # 采用的损失函数
        "booster":"gbtree",           # 采用梯度提升树模型
        "silent":0,                   # 不打印信息
        "nthread":3,                  # 训练用的线程数
        'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'eta': 0.007,  # 如同学习率
        "min_child_weight":2,
        "subsample":0.6,
        "colsample_bytree":0.5,
        "scale_pos_weight":0.3
    }

    evals=[(dtrain,"train"),(dvalid,"val")]
    bst = XGBRegressor(base_score=0.5,
                       colsample_bylevel=1,
                       colsample_bytree=1,
                       gamma=0,
                       learning_rate=0.1,
                       max_delta_step=0,
                       max_depth=3,
                       min_child_weight=1,
                       missing=None,
                       n_estimators=100,
                       nthread=3,
                       objective="reg:linear",
                       reg_alpha=0,
                       reg_lambda=1,
                       scale_pos_weight=1,
                       seed=1850,
                       silent=True,
                       subsample=1
                       )
    # modelfit(bst,traindata,traintarget,validdata,validtarget)

    # 使用GridSearchCV进行调参数
    testparams={
        "max_depth":list(range(3,10,2)),
        "min_child_weight":list(range(1,6,2)),
        "n_estimators":list(range(50,200,10))
    }
    gsearch=GridSearchCV(estimator=bst,param_grid=testparams,scoring='r2',cv=5)

    gsearch.fit(traindata,traintarget)
    predictdata=gsearch.predict(validdata)
    # gsearch.score()

    if not os.path.exists('model/'):
        os.makedirs('model/')
    # bst.save_model('model/test.model')

"""Xgboost模型预测"""
def predict():
    param={
        "objective":"reg:logistic",   # 回归问题
        "eval_metric":"rmse",         # 采用的损失函数
        "booster":"gbtree",           # 采用梯度提升树模型
        "silent":1,                   # 不打印信息
        "nthread":3,                  # 训练用的线程数
        'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'eta': 0.007,  # 如同学习率
        "min_child_weight":2,
        "subsample":0.6,
        "colsample_bytree":0.5,
        "scale_pos_weight":0.3
    }
    bst=xgb.Booster(params=param)# 初始化模型
    bst.load_model('model/test.model')


if __name__ == '__main__':
    xgboostF()