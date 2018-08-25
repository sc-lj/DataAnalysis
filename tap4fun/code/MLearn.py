# coding:utf-8

"""
该代码是给机器学习提供特征的。
"""

try:
    from code.Script import *
except:
    from Script import *
import pandas as pd
import numpy as np
import copy,time,random,math,re
from sklearn.preprocessing import *
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.utils import shuffle
import xgboost as xgb
from xgboost import XGBRegressor,XGBClassifier
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
import os


scaler=StandardScaler()
MinMax=MinMaxScaler(feature_range=(0,250))
normal=Normalizer()

def check_Row(files):
    """
    查看训练数据中数据自变量中全是0和因变量是否大于0的关系,下面统计的变量不包括avg_online_minutes(平均在线时间)这个变量。
    自变量都是零的时候，因变量也是0的个数有607047个；自变量都是零的时候，因变量不是0的个数有0个。
    当非零的自变量个数有1个的时候，因变量是0的个数有168492个；因变量不是0的个数有0个。
    当非零的自变量个数有2个的时候，因变量是0的个数有10555个；因变量不是0的个数有1392个。
    当非零的自变量个数有3个的时候，因变量是0的个数有10865个；因变量不是0的个数有177个。
    当非零的自变量个数有4个的时候，因变量是0的个数有59627个；因变量不是0的个数有130个。
    :param files:
    :return:
    """
    data=pd.read_csv(files,index_col=0)
    index=[]
    indexs=data.index
    i=0
    num=2
    zero_num = 0
    nonzero_num = 0
    for da in data.values:
        da=list(da)[1:-1]
        del da[-3]
        newda=list(set(da))
        label=da[-1]
        if len(newda)<=num:
            if label==0:
                zero_num+=1
                index.append(indexs[i])
            else:nonzero_num+=1
        i+=1
    print("自变量至少有%s个不为零的时候，因变量为0的个数"%str(num-1),zero_num)
    print("自变量至少有%s个不为零的时候，因变量不为0的个数"%str(num-1), nonzero_num)
    num+=1
    data.drop(index,inplace=True)
    # print(data.shape)
    data.to_csv(drop_zero)
    return data

def classify(files):
    """
    将每个样本中，变量值大于0的改为1。这可以用来分类。
    :return:
    """
    data = pd.read_csv(files, index_col=0)
    data.drop("register_time",inplace=True,axis=1)
    for var in data:
        data[var]=data[var].apply(lambda x:1 if x>0 else 0,axis=0)
    data.to_csv(classifyfile)


def addColumn(files):
    """
    添加一列，该列统计了每个样本有多少个非零的变量（除avg_online_minutes(平均在线时间)这个变量之外）。
    再新增一列，该列是对样本进行类别分类，后45天有付费的标签为1，没有付费的为0
    :param files:
    :return:
    """
    data = pd.read_csv(files, index_col=0)
    print(data.shape)
    columns = data.columns.tolist()
    columns.insert(107,"nonzeronum")
    columns.insert(108, "category")  # 该列是对样本再次进行分类，后45天有付费的标签为1，没有为0。

    newvalue=[]
    for da in data.values:
        da=list(da)[1:-1]
        del da[-3]
        newda=[i for i in da if i>0]
        num=len(newda)
        newvalue.append(num)

    data=data.reindex(columns=columns)
    data["category"]=data['prediction_pay_price'].apply(lambda x:0 if x<=0 else 1)
    data['nonzeronum']=newvalue
    print(data.shape)
    data.to_csv(files,float_format="%.5f")


def analysis_level_var(files,isstd=True):
    """
    :param files:
    :param isstd: 是否按照标准差进行数据标准化
    :return:
    """
    data=pd.read_csv(files,index_col=0,parse_dates=['register_time'])
    # ～是取相反的意思，判断是否都是数字，占用内存很大，不划算
    # new=data[~data.applymap(np.isreal).all(1)]
    # new =np.argmin(chunk.applymap(np.isreal).all(1))
    data.rename(columns={'treatment_acceleraion_add_value':'treatment_acceleration_add_value'},inplace=True)
    print(data.shape)
    variable=data.columns.tolist()
    depvar=variable[-2:]#因变量有category,prediction_pay_price
    new_data=pd.DataFrame()
    X_var=variable[:-2]

    for cate in category:
        var_add=cate+category_suffix[0]
        var_reduce=cate+category_suffix[1]
        del X_var[X_var.index(var_add)]
        del X_var[X_var.index(var_reduce)]
        # 将获取的数量减去消耗的数量
        new_data[cate]=data[var_add]-data[var_reduce]
        # 并将消耗的数量一并存入到新数据中
        new_data[var_reduce]=data[var_reduce]

    # 对值变动范围很大的物品进行标准化
    if isstd:
        std_data = scaler.fit_transform(new_data)
    else:
        #将值压缩到一定范围
        std_data=MinMax.fit_transform(new_data)
    std_data = pd.DataFrame(std_data, index=new_data.index, columns=new_data.columns)

    X=data[X_var]
    X=pd.concat([X,std_data],axis=1)
    columns = X.columns.tolist()
    levelvar = [column for column in columns if re.match('.+_level', column)]

    # 根据变量的方差降维
    invar = columns[1:]
    othervar=columns[:1]
    invarvalue=X[invar]
    describe=invarvalue.std()
    # 删除了自变量的方差小于0.1的变量。考虑到里面的level都是有序变量，所以方差设置的比较小
    # 当方差阈值设置为0.5的时候，删除了45个变量，这些特征不发散。
    var=describe[describe.apply(lambda x:True if x>0.5 else False)].index

    varvalue=X[var]
    othervalue=X[othervar]

    Y=data[depvar]
    data=pd.concat([othervalue,varvalue,Y],axis=1)
    # decimals_var = X.columns.tolist()[1:]
    # decimals=pd.Series([5]*len(decimals_var),index=decimals_var)
    print(data.shape)
    data.to_csv(tapfun,float_format="%.5f")


def cut_data(filename):
    """
    切割文件，生成训练集和测试集，将整个数据切分成4部分，
    总共有2288007个样本
    1、后45天有付费记录且前7天也有付费记录，这类有41439个样本
    2、后45天没有付费记录但是前7天有付费记录，根据数据这类人为0，
    3、后45天没有付费记录但前7天有付费记录，这类有4549个样本
    4、后45天没有付费记录且前7天也没有付费记录，这类有2242019个样本
    这些数据按照10%的比例切分。
    :param filename: 文件名
    :return:
    """
    data=pd.read_csv(filename,index_col=0)
    data1=data[(data['prediction_pay_price']>0)&(data['pay_price']>0)]
    new_data1=data1.sample(frac=0.1)
    # data2 = data[(data['prediction_pay_price'] <=0) & (data['pay_price'] > 0)]
    # new_data2=data2.sample(frac=0.1)
    data3 = data[(data['prediction_pay_price'] > 0) & (data['pay_price'] <= 0)]
    new_data3=data3.sample(frac=0.1)
    data4 = data[(data['prediction_pay_price'] <= 0) & (data['pay_price'] <= 0)]
    new_data4=data4.sample(frac=0.1)
    new_data=pd.concat([new_data1,new_data3,new_data4],axis=0)
    new_data=shuffle(new_data)
    new_data.to_csv(valid_file,float_format = '%.5f')
    df=data.drop(list(new_data.index),axis=0)
    df=shuffle(df)
    df.to_csv(train_file,float_format = '%.5f')#让输出的浮点数的精度为5。


def checkColumn(files):
    """
    检查每个变量包含有多少个不一样的值。
    :param files:
    :return:
    """
    data = pd.read_csv(files, index_col=0)
    columns=data.columns
    for column in columns:
        da=data[column]
        if len(set(da))<=2:
            print(column)
            value=list(set(da))[-1]
            print(len(data[data[column]==value]))


def getFeature(files,istrain=True):
    data=pd.read_csv(files,index_col=0)
    columns=data.columns.tolist()
    invar=columns[1:-2]
    devar=columns[-1]
    levelvar=[column for column in  columns if re.match('.+_level',column)]

    varvalue=np.array(data[invar].values)
    target=np.array(data[devar].values)
    return varvalue,target


def DealTestData(files):
    """

    :param file:
    :return:
    """
    data=pd.read_csv(files,index_col=0)
    # 删除训练数据中数据自变量中只有一个自变量不为0的样本, 下面统计的变量不包括avg_online_minutes(平均在线时间)这个变量。
    zero_num=0
    index=[]
    indexs=data.index
    i=0
    for da in data.values:
        da=list(da)[1:]
        del da[-3]
        newda=list(set(da))
        if len(newda)<=2:
            zero_num+=1
            index.append(indexs[i])
        i+=1
    print(zero_num)
    data.drop(index,inplace=True)

    # 添加一列，该列统计了每个样本有多少个非零的变量（除avg_online_minutes(平均在线时间)这个变量之外）
    print(data.shape)
    columns = data.columns.tolist()
    columns.insert(107, "nonzeronum")
    newvalue = []
    for da in data.values:
        da = list(da)[1:]
        del da[-3]
        newda = [i for i in da if i > 0]
        num = len(newda)
        newvalue.append(num)

    data = data.reindex(columns=columns)
    data['nonzeronum'] = newvalue

    print(data.shape)
    data.rename(columns={'treatment_acceleraion_add_value': 'treatment_acceleration_add_value'}, inplace=True)
    X_var= data.columns.tolist()
    new_data = pd.DataFrame()

    for cate in category:
        var_add = cate + category_suffix[0]
        var_reduce = cate + category_suffix[1]
        del X_var[X_var.index(var_add)]
        del X_var[X_var.index(var_reduce)]
        # 将获取的数量减去消耗的数量
        new_data[cate] = data[var_add] - data[var_reduce]
        # 并将消耗的数量一并存入到新数据中
        new_data[var_reduce] = data[var_reduce]

    # 对值变动范围很大的物品进行标准化
    # std_data = scaler.fit_transform(new_data)
    # 将值压缩到一定范围
    std_data = MinMax.fit_transform(new_data)
    std_data = pd.DataFrame(std_data, index=new_data.index, columns=new_data.columns)

    X = data[X_var]
    data = pd.concat([X, std_data], axis=1)

    print(data.shape)

    data.to_csv('../data/TapFunTest.csv', float_format="%.5f")


if __name__ == '__main__':
    # analysis_resource_var(drop_zero)
    # checkColumn('../data/drop_zero.csv')

    """按照下面顺序执行"""
    # 删除行
    check_Row(tap_fun_train)
    # 增加列
    # addColumn(drop_zero)
    # 标准化
    # analysis_level_var(drop_zero)
    # 切分数据集
    # cut_data(tapfun)

    # getFeature(tapfun)
    # getFeature(valid_file)
    # getFeature(train_file)

    # DealTestData(tap_fun_test)

