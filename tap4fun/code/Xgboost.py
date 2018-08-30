# coding:utf-8
try:
    from code.Script import *
    from code.MLearn import *
except:
    from Script import *
    from MLearn import *

import xgboost as xgb
from xgboost import XGBRegressor,XGBClassifier
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib


"""训练Xgboost 回归模型"""
def xgboostF():
    traindata, traintarget=getFeature(train_file)
    validdata,validtarget=getFeature(valid_file)
    # 加载array格式的数据到xgb中。
    dtrain = xgb.DMatrix(traindata, label=traintarget)
    dvalid=xgb.DMatrix(validdata,label=validtarget)


    bst = XGBRegressor(base_score=0.5,
                       colsample_bylevel=1,
                       colsample_bytree=1,
                       gamma=0,
                       learning_rate=0.1,# 学习率
                       max_delta_step=0,
                       max_depth=3,
                       min_child_weight=1,
                       missing=None,
                       n_estimators=500,# 迭代次数
                       nthread=3,
                       objective="reg:linear",
                       reg_alpha=0,
                       reg_lambda=1,
                       scale_pos_weight=1,
                       seed=1850,
                       silent=True,
                       subsample=1,

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


# https://segmentfault.com/a/1190000014040317
# noinspection PyDeprecation
def xgboostparams():
    X_train, Y_train=read_data(train_file)
    # validdata,validtarget=getFeature(valid_file)
    # cv_params = {'n_estimators': [200, 250, 300, 350, 400]}
    # cv_params={'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
    cv_params = {'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
    other_params = {'learning_rate': 0.1,# 学习率
                    'n_estimators': 400, # 迭代次数
                    'max_depth': 7,# 深度
                    'min_child_weight': 6, #子节点的权重
                    'seed': 0,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 0,
                    'reg_alpha': 0,
                    'reg_lambda': 1}
    model = XGBRegressor(**other_params)
    # 调参数
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(X_train, Y_train)
    # evalute_result = optimized_GBM.grid_scores_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('the best params is :{0}'.format(optimized_GBM.best_params_))
    print('the model best score is :{0}'.format(optimized_GBM.best_score_))


def read_data(files):
    train=pd.read_csv(files,index_col=0)
    columns=train.columns.tolist()
    dta=train[columns[:-1]]
    label=train[columns[-1]]
    return dta,label


def xgboostLinear():
    X_train, Y_train=read_data(train_file)
    X_test,Y_test=read_data(valid_file)
    # XGBoost训练过程，下面的参数就是刚才调试出来的最佳参数组合
    model = xgb.XGBRegressor(learning_rate=0.1, n_estimators=400, max_depth=7, min_child_weight=6, seed=0,
                             subsample=0.7, colsample_bytree=0.7, gamma=0.1, reg_alpha=1, reg_lambda=1)
    model.fit(X_train, Y_train)
    if not os.path.exists("./model/xgboost"):
        os.makedirs("./model/xgboost")

    joblib.dump(model,"./model/xgboost/xgboost.m")
    # 对测试集进行预测
    ans = model.predict(X_test)
    print('R-squared value of Poly SVR is', model.score(X_test, Y_test))
    print('The root mean squared error of Poly SVR is', np.sqrt(mean_squared_error(Y_test, ans)))

def xgboostPredict(testfile):
    data=pd.read_csv(testfile,index_col=0)
    columns = data.columns.tolist()
    model=joblib.load("./model/xgboost/xgboost.m")
    anc=model.predict(data[columns])
    df=pd.DataFrame(anc,columns=["prediction_pay_price"],index=data.index)
    df.to_csv("../data/predict.csv")




if __name__ == '__main__':
    xgboostparams()
    # xgboostLinear()
    # xgboostPredict("../data/TapFunTest.csv")
