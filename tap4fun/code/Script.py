# coding:utf-8

        # 木头获取数量,木头消耗数量,石头获取数量,石头消耗数量,象牙获取数量,象牙消耗数量,肉获取数量,肉消耗数量
indepvar={"资源":['wood_add_value','wood_reduce_value','stone_add_value','stone_reduce_value','ivory_add_value','ivory_reduce_value','meat_add_value','meat_reduce_value'],

      # 科研：木材生产，科研：石头生产，科研：象牙生产，科研：肉类生产,科研：魔法生产
      "资源生产": ['sr_rss_a_prod_levell', 'sr_rss_b_prod_level', 'sr_rss_c_prod_level', 'sr_rss_d_prod_level','sr_rss_e_prod_level',
               # 科研：木材采集,科研：石头采集,科研：象牙采集,科研：肉类采集,科研：魔法采集
               'sr_rss_a_gather_level', 'sr_rss_b_gather_level', 'sr_rss_c_gather_level', 'sr_rss_d_gather_level','sr_rss_e_gather_level'],

    #         魔法获取数量,魔法消耗数量
    "存活技能":['magic_add_value','magic_reduce_value'],

    #         勇士招募数量, 勇士损失数量,  驯兽师招募数量, 驯兽师损失数量, 萨满招募数量, 萨满损失数量,
    "创建团队":["infantry_add_value",'infantry_reduce_value','cavalry_add_value','cavalry_reduce_value','shaman_add_value','shaman_reduce_value'],

    #            勇士伤兵产生数量,勇士伤兵恢复数量, 驯兽师伤兵产生数量,驯兽师伤兵恢复数量,萨满伤兵产生数量,萨满伤兵恢复数量,
    "团队恢复能力":['wound_infantry_add_value','wound_infantry_reduce_value','wound_cavalry_add_value','wound_cavalry_reduce_value','wound_shaman_add_value','wound_shaman_reduce_value'],

    #         通用加速获取数量,通用加速使用数量,建筑加速获取数量,建筑加速使用数量
    "辅助技能":['general_acceleration_add_value','general_acceleration_reduce_value','building_acceleration_add_value','building_acceleration_reduce_value',
            # 科研加速获取数量,科研加速使用数量,训练加速获取数量,训练加速使用数量
            'reaserch_acceleration_add_value','reaserch_acceleration_reduce_value','training_acceleration_add_value','training_acceleration_reduce_value',
            # 治疗加速获取数量,治疗加速使用数量
            'treatment_acceleration_add_value','treatment_acceleration_reduce_value'],

    # 建筑：士兵小屋等级，建筑：治疗小井等级，建筑：要塞等级，建筑：据点传送门等级，建筑：兵营等级，
    "辅助技能等级":['bd_training_hut_level','bd_healing_lodge_level','bd_stronghold_level','bd_outpost_portal_level','bd_barrack_level',
        #建筑：治疗之泉等级,建筑：智慧神庙等级,建筑：联盟大厅等级,建筑：仓库等级,建筑：瞭望塔等级,
        'bd_healing_spring_level','bd_dolmen_level','bd_guest_cavern_level','bd_warehouse_level','bd_watchtower_level',
        # 建筑：魔法幸运树等级,建筑：战争大厅等级,建筑：联盟货车等级,建筑：占卜台等级,建筑：祭坛等级,
        'bd_magic_coin_tree_level','bd_hall_of_war_level','bd_market_level','bd_hero_gacha_level','bd_hero_strengthen_level',
        # 建筑：冒险传送门等级,科研：侦查等级,科研：训练速度等级
        'bd_hero_pve_level','sr_scout_level','sr_training_speed_level'],

    #   科研：建造速度
    "建造速度": ['sr_construction_speed_level'],

    #        科研：守护者,科研：巨兽驯兽师,科研：吟唱者
    "成员等级":['sr_infantry_tier_2_level','sr_cavalry_tier_2_level','sr_shaman_tier_2_level',
              # 科研：战斗大师,科研：高阶巨兽骑兵,科研：图腾大师
              'sr_infantry_tier_3_level','sr_cavalry_tier_3_level','sr_shaman_tier_3_level',
              # 科研：狂战士,科研：龙骑兵,科研：神谕者
              'sr_infantry_tier_4_level', 'sr_cavalry_tier_4_level', 'sr_shaman_tier_4_level'],

    #            科研：勇士防御,科研：驯兽师防御,科研：萨满防御,
    "成员防御等级":['sr_infantry_def_level','sr_cavalry_def_level','sr_shaman_def_level',
              # 科研：勇士生命,科研：驯兽师生命,科研：萨满生命
              'sr_infantry_hp_level','sr_cavalry_hp_level','sr_shaman_hp_level'],

    #     科研：部队攻击,科研：部队防御,科研：部队负重,科研：部队消耗
    "troop":['sr_troop_attack_level','sr_troop_defense_level','sr_troop_load_level','sr_troop_consumption_level'],

    # 科研：勇士攻击,科研：驯兽师攻击,科研：萨满攻击
    "成员攻击等级":['sr_infantry_atk_level','sr_cavalry_atk_level','sr_shaman_atk_level'],

    # 科研：资源保护，科研：医院容量,科研：治疗速度
    "":['sr_hide_storage_level','sr_healing_space_level','sr_healing_speed_level'],

    #     科研：据点耐久,科研：据点二,科研：据点三,科研：据点四
    "据点":['sr_outpost_durability_level','sr_outpost_tier_2_level','sr_outpost_tier_3_level','sr_outpost_tier_4_level'],

    #     科研：联盟行军速度,科研：战斗行军速度,科研：采集行军速度
    "速度":['sr_alliance_march_speed_level','sr_pvp_march_speed_level','sr_gathering_march_speed_level'],

    #     科研：增援部队容量,科研：行军大小,科研：资源帮助容量
    "help":['sr_guest_troop_capacity_level','sr_march_size_level','sr_rss_help_bonus_level'],

    #     PVP次数,主动发起PVP次数,PVP胜利次数 ；  "PVP，指玩家对战玩家（Player versus player），即玩家互相利用游戏资源攻击而形成的互动竞技。"
    "pvp":['pvp_battle_count','pvp_lanch_count','pvp_win_count'],

    #     PVE次数,主动发起PVE次数,PVE胜利次数；   "PVE：玩家对战环境"
    "pve":['pve_battle_count','pve_lanch_count','pve_win_count'],

    # 在线时长，付费金额，付费次数
    "吸引力":['avg_online_minutes','pay_price','pay_count'],

    # 科研：领土采集奖励,
    "other":['sr_gathering_hunter_buff_level']
    }

# 45日付费金额
Depvar='prediction_pay_price'


def gen_invar():
    Indepvar=[]
    for key,value in indepvar.items():
        Indepvar.extend(value)
    return Indepvar

# 选择45日付费金额大于0或者7日付费金额大于0
vip_file='../data/vip.csv'
# 选择45日付费金额小于等于0或者7日付费金额小于等于0
nonvip_file='../data/nonvip.csv'
# 选择45日付费金额小于等于0并且7日付费金额小于等于0
revert_file='../data/revert.csv'
# 总训练数据集
tap_fun_train='../data/tap_fun_train.csv'
# 测试数据集
tap_fun_test='../data/tap_fun_test.csv'

# 训练数据集
train_file='../data/train.csv'

# 验证数据集
valid_file='../data/valid.csv'

# 去掉了自变量中全是0和因变量也是0的样本
drop_zero='../data/drop_zero.csv'

# 剔除了一些自变量，又增加一些自变量
tapfun='../data/tapfun.csv'


# 这些都是有相同后缀的变量,_add_value,_reduce_value
category=['wood','stone','ivory','meat','magic','infantry','cavalry','shaman','wound_infantry','wound_cavalry','wound_shaman',
          'general_acceleration','building_acceleration','reaserch_acceleration','training_acceleration','treatment_acceleration',
          ]
# 后缀
category_suffix=['_add_value','_reduce_value']


import argparse
def argument():
    args=argparse.ArgumentParser()
    args.add_argument('--filter',default=4,type=int,help='卷积核的大小,也是pooling的大小')
    args.add_argument('--filter_num',default=20,type=int,help='卷积核的个数')
    args.add_argument('--epochs',default=100,type=int,help='循环次数')
    args.add_argument('--batch_size',default=100,type=int,help='每次训练块大小')
    args.add_argument('--stride',default=1,type=int,help='卷积核移动的步长')
    args.add_argument('--dropout',default=0.5,type=float,help='dropout 概率')
    args.add_argument('--lr',default=0.1,type=float,help='学习率')
    args.add_argument('--out_dir',default='./',type=str,help='结果输出文件夹')
    args.add_argument('--max_checkpoints',default=3,type=int,help='模型结果最大保存的数量')
    args.add_argument('--evaluate_every',default=100,type=int,help='每训练多少轮，就进行验证')

    arg=args.parse_args()
    return arg
import logging

def log_config(name):
    logger=logging.getLogger(name)
    logger.setLevel(logging.INFO)
    sh=logging.FileHandler('./log.log')
    fmt='%(asctime)s %(filename)s %(funcName)s %(lineno)s line %(levelname)s >>%(message)s'
    dtfmt='%Y %m %d %H:%M:%S'
    formatter=logging.Formatter(fmt=fmt,datefmt=dtfmt)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger