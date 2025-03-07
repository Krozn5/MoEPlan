import numpy as np
import os
import torch
import torch.nn as nn
import time
import pandas as pd
from scipy.stats import pearsonr

from model.util import Normalizer
from model.database_util import get_hist_file, get_job_table_sample, collator
from model.model import QueryFormer
from model.database_util import Encoding
from model.dataset import PlanTreeDataset,PlanTreeMoEDataset

import os


def txt_to_df(f_paths,aim_expert = None,job_light_path = False,synthetic_path = False,tpcds1gb_path = False):
    #  30 = 21
    num_experts = len(aim_expert)
    my_dict1 = dict()
    my_str = None
    temp = 0
    for f_path in f_paths:
        with open(f_path) as f:
            state = False
            for line in f:
                pos_index = line.find('hint set ')
                pos = line.find('sql ')
                if(job_light_path == True):
                    pos1 = pos-3
                    pos2 = pos+4
                elif(synthetic_path == True):
                    pos1 = pos-5
                    pos2 = pos+4
                elif(tpcds1gb_path == True):
                    pos1 = pos-4
                    pos2 = pos+4
                else:
                    return False
                if(pos==-1 and pos_index==-1):
                    continue
                if(pos_index != -1):
                    state = False
                    pos_index = pos_index + 9
                    # print(line[pos_index:pos_index+2])
                    if(int(line[pos_index:pos_index+2]) in aim_expert):
                        state = True
                        if(line[pos_index+1]==' '):
                            index = line[pos_index]
                        else:
                            index = line[pos_index:pos_index+2]
                        my_dict1[index] = {}
                        my_str = index
                if(state == False):
                    continue
                if(pos!=-1):
                    pos = pos+14    
                    tmp_name = line[pos1:pos2-1]
                    tmp_time = line[pos2:-4]
                    if(tmp_name[0] == ' '):
                        tmp_name = tmp_name[1:]
                    if(tmp_time[-1]==' '):
                        tmp_time = tmp_time[:-1]
                    my_dict1[my_str][tmp_name] = tmp_time
    to_df = pd.DataFrame.from_dict(my_dict1)
    # print(to_df.iloc[0,:])
    to_df.columns = [str(i) for i in range(to_df.shape[1])]
    to_df = to_df.iloc[:,:num_experts]
    # to_df.columns = [str(i) for i in range(num_experts)]
    to_df = to_df.astype('float')
    to_df[str(num_experts)] = to_df.min(axis=1)
    to_df[str(num_experts+1)] = to_df.apply(lambda x: x.values.argmin(), axis=1)
    # to_df = pd.DataFrame({'y_pred': to_df['0'].apply(lambda x: x[:num_experts])})
    to_df = pd.DataFrame({'y': to_df.apply(lambda x: x.tolist(), axis=1)})
    to_df = pd.DataFrame({'y_pred': to_df['y'].apply(lambda x: x[:num_experts+2])})
    to_df = to_df.sort_index()
    to_df.insert(loc=0, column='id', value=[i for i in range(len(to_df))])
    '''The total number of items is num_experts+2, where the num_experts-th item represents 
    the min_value, and the num_experts+1-th item represents the coordinate of the minimum.'''
    return to_df

def df_to_ds(database,f_path,df,methods):
    methods = methods
    get_table_sample = methods['get_sample']
    workload_file_name = './data/'+database+'/workloads/' + f_path
    table_sample = get_table_sample(workload_file_name)
    plan_df = pd.read_csv('./data/'+database+'/{}_plan.csv'.format(f_path))
    df_merge = pd.merge(df,plan_df,on='id')
    return PlanTreeMoEDataset(df_merge, None, \
    methods['encoding'], methods['hist_file'], methods['cost_norm'], \
    methods['cost_norm'], 'cost', table_sample)

def get_dataset(args):
    # args.dataset = 'tpcds'
    if(args.dataset == 'job'):
        data_path = './data/job/'
        database='job'
        workload='job'
        f_path = ['./data/job.txt']
        aim_expert = [i for i in range(args.num_experts)]
    elif(args.dataset == 'tpcds'):
        data_path = './data/tpcds1gb/'
        database = 'tpcds1gb'
        workload = 'synthetic'
        f_path = ['./data/tpcds1gb.txt']
        aim_expert = [i for i in range(args.num_experts)]
    elif(args.dataset == 'tpcds_new'):
        data_path = './data/tpcds_new/'
        database = 'tpcds_new'
        workload = 'tpcdssplit'
        f_path = ['./data/tpcdssplit.txt']
        aim_expert = [i for i in range(args.num_experts)]
    else:
        print("wrong")
        print(z)# break

    hist_file = get_hist_file(data_path + 'histogram_string.csv')
    cost_norm = args.cost_norm
    card_norm = args.card_norm

    column_min_max_file = pd.read_csv(data_path + 'column_min_max_vals.csv')
    column_min_max_vals, col2idx = {}, {}
    col_idx = 0
    for index, row in column_min_max_file.iterrows():
        column_min_max_vals[row['name']] = (row['min'], row['max'])
        col2idx[row['name']] = col_idx
        col_idx += 1
    col2idx['NA'] = col_idx
    col_idx += 1
    oplist = ['>=', '<=', '!=', '<>', '~~', '>', '<', '=', 'NA']
    op2idx = dict(zip(oplist, list(range(len(oplist)))))
    idx2op = dict(zip(list(range(len(oplist))), oplist))
    encoding = Encoding(column_min_max_vals, col2idx, op2idx, idx2op)
    methods = {
        'get_sample' : get_job_table_sample,
        'encoding': encoding,
        'cost_norm': cost_norm,
        'hist_file': hist_file,
        'model': None,
        'device': args.device,
        'bs': args.bs,
    }  
    get_table_sample = methods['get_sample']
    tpcds_df= txt_to_df(f_path,aim_expert,tpcds1gb_path= True)
    if(args.dataset == 'job'):
        ds_tpcds1gb= df_to_ds('job','job',tpcds_df,methods)
    elif(args.dataset == 'tpcds'):
        ds_tpcds1gb= df_to_ds('tpcds1gb','synthetic',tpcds_df,methods)
    elif(args.dataset == 'tpcds_new'):
        ds_tpcds1gb= df_to_ds('tpcds_new','tpcdssplit',tpcds_df,methods)
    return tpcds_df,ds_tpcds1gb
