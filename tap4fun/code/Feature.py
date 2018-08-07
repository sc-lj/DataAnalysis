# coding:utf-8

"""
该代码是给深度学习提供训练数据的。
"""
import os,sys
try:
    from code.Script import *
except:
    from Script import *
import pandas as pd
import numpy as np
import copy
from sklearn.utils import shuffle

def change_loc(vector,num,all):
    """
    交换变量的位置
    :param vector:[2,3,4,5,6]
    :param num: 2
    :param all: len(vector)
    :return: [4,5,6,2,3]
    """
    vecto=copy.deepcopy(vector)
    new=vecto[0:num]
    vecto.extend(new)
    return vecto[-all:]


def feature_matrix(vector,num=4):
    """
    对每个样本生成一个矩阵
    :param vector:样本向量
    :param num:滑动的变量个数
    :return:
    """
    length=len(vector)
    new_vector=[]
    for i in range(0,length,num):
        new_vector.append(change_loc(vector,i+num,length))
    return new_vector

arg=argument()

def read_batch(batch_data,predict=False):
    batch_matrix=[]
    labels=[]
    for i in range(batch_data.shape[0]):
        if not predict:
            newda = list(batch_data.iloc[i].values[1:-2])
            label=[batch_data.iloc[i].values[-2]]
            labels.append(label)
        else:
            newda = list(batch_data.iloc[i].values[1:])
        matrix=feature_matrix(newda,arg.filter)
        batch_matrix.append(matrix)
    return batch_matrix,labels

def shuff(files):
    data=pd.read_csv(files,index_col=0)
    data = shuffle(data)
    data.to_csv(files,float_format='%.5f')

def gen_batch(files,batch,predict=False):
    """
    迭代文件数据
    :param files: 文件名
    :param batch: 块大小
    :return:
    """
    if not predict:
        shuff(files)
    data=pd.read_csv(files,index_col=0,chunksize=batch)
    for da in data:
        matrixs,labels=read_batch(da,predict)
        matrixs=np.array(matrixs)
        print(matrixs.shape)
        labels=np.array(labels)
        yield matrixs,labels

def gen_train(files,batch,log):
    for epoch in range(arg.epochs):
        log.info('epoch step {}'.format( epoch))
        for matrixs,labels in gen_batch(files,batch):
            yield matrixs,labels





if __name__ == '__main__':
    vector=['wood_add_value', 'wood_reduce_value', 'stone_add_value', 'stone_reduce_value', 'ivory_add_value', 'ivory_reduce_value', 'meat_add_value', 'meat_reduce_value', 'sr_rss_a_prod_levell', 'sr_rss_b_prod_level', 'sr_rss_c_prod_level', 'sr_rss_d_prod_level', 'sr_rss_e_prod_level', 'sr_rss_a_gather_level', 'sr_rss_b_gather_level', 'sr_rss_c_gather_level', 'sr_rss_d_gather_level', 'sr_rss_e_gather_level', 'magic_add_value', 'magic_reduce_value', 'infantry_add_value', 'infantry_reduce_value', 'cavalry_add_value', 'cavalry_reduce_value', 'shaman_add_value', 'shaman_reduce_value', 'wound_infantry_add_value', 'wound_infantry_reduce_value', 'wound_cavalry_add_value', 'wound_cavalry_reduce_value', 'wound_shaman_add_value', 'wound_shaman_reduce_value', 'general_acceleration_add_value', 'general_acceleration_reduce_value', 'building_acceleration_add_value', 'building_acceleration_reduce_value', 'reaserch_acceleration_add_value', 'reaserch_acceleration_reduce_value', 'training_acceleration_add_value', 'training_acceleration_reduce_value', 'treatment_acceleration_add_value', 'treatment_acceleration_reduce_value', 'bd_training_hut_level', 'bd_healing_lodge_level', 'bd_stronghold_level', 'bd_outpost_portal_level', 'bd_barrack_level', 'bd_healing_spring_level', 'bd_dolmen_level', 'bd_guest_cavern_level', 'bd_warehouse_level', 'bd_watchtower_level', 'bd_magic_coin_tree_level', 'bd_hall_of_war_level', 'bd_market_level', 'bd_hero_gacha_level', 'bd_hero_strengthen_level', 'bd_hero_pve_level', 'sr_scout_level', 'sr_training_speed_level', 'sr_construction_speed_level', 'sr_infantry_tier_2_level', 'sr_cavalry_tier_2_level', 'sr_shaman_tier_2_level', 'sr_infantry_tier_3_level', 'sr_cavalry_tier_3_level', 'sr_shaman_tier_3_level', 'sr_infantry_tier_4_level', 'sr_cavalry_tier_4_level', 'sr_shaman_tier_4_level', 'sr_infantry_def_level', 'sr_cavalry_def_level', 'sr_shaman_def_level', 'sr_infantry_hp_level', 'sr_cavalry_hp_level', 'sr_shaman_hp_level', 'sr_troop_attack_level', 'sr_troop_defense_level', 'sr_troop_load_level', 'sr_troop_consumption_level', 'sr_infantry_atk_level', 'sr_cavalry_atk_level', 'sr_shaman_atk_level', 'sr_hide_storage_level', 'sr_healing_space_level', 'sr_healing_speed_level', 'sr_outpost_durability_level', 'sr_outpost_tier_2_level', 'sr_outpost_tier_3_level', 'sr_outpost_tier_4_level', 'sr_alliance_march_speed_level', 'sr_pvp_march_speed_level', 'sr_gathering_march_speed_level', 'sr_guest_troop_capacity_level', 'sr_march_size_level', 'sr_rss_help_bonus_level', 'pvp_battle_count', 'pvp_lanch_count', 'pvp_win_count', 'pve_battle_count', 'pve_lanch_count', 'pve_win_count', 'avg_online_minutes', 'pay_price', 'pay_count', 'sr_gathering_hunter_buff_level']
    # feature_matrix(vector)
    # read_csv('../data/revert.csv')
    # gen_batch('../data/revert.csv',6)
    qa,_=gen_batch(train_file,8,predict=False)
    qa.__next__()


