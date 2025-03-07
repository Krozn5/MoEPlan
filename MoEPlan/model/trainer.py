import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .dataset import PlanTreeDataset
from .database_util import collator, get_job_table_sample
import os
import time
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from tqdm import tqdm
import json
import random
# from model.model_Mebo import gate_loss
def gate_loss(gating_weights, lambda_balance=0.1):
        balance_loss = torch.std(gating_weights.sum(0))
        return lambda_balance*balance_loss


def retrain_threshold(t_select_min,t_db):
    x = (t_select_min.squeeze() > t_db.squeeze())

    return x.sum(),torch.nonzero(x,as_tuple=False)[:,0]


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def print_multi_qerror(pos_preds_unnorm, pos_labels_unnorm, cost_preds_unnorm, cost_labels_unnorm, prints=False):
    qerror = []
    for i in range(len(pos_preds_unnorm)):
        if pos_preds_unnorm[i] > float(pos_labels_unnorm[i]):
            qerror.append(pos_preds_unnorm[i] / float(pos_labels_unnorm[i]))
        else:
            qerror.append(float(pos_labels_unnorm[i]) / float(pos_preds_unnorm[i]))

    e_50, e_90 = np.median(qerror), np.percentile(qerror,90)    
    e_mean = np.mean(qerror)

    if prints:
        print("Pos Median: {}".format(e_50))
        print("Pos Mean: {}".format(e_mean))

    pos_res = {
        'q_median' : e_50,
        'q_90' : e_90,
        'q_mean' : e_mean,
    }
    
    qerror = []
    for i in range(len(cost_preds_unnorm)):
        if cost_preds_unnorm[i] > float(cost_labels_unnorm[i]):
            qerror.append(cost_preds_unnorm[i] / float(cost_labels_unnorm[i]))
        else:
            qerror.append(float(cost_labels_unnorm[i]) / float(cost_preds_unnorm[i]))

    e_50, e_90 = np.median(qerror), np.percentile(qerror,90)    
    e_mean = np.mean(qerror)

    if prints:
        print("Cost Median: {}".format(e_50))
        print("Cost Mean: {}".format(e_mean))

    cost_res = {
        'q_median' : e_50,
        'q_90' : e_90,
        'q_mean' : e_mean,
    }

    return {'pos': pos_res, 'cost': cost_res}

def print_qerror(preds_unnorm, labels_unnorm, prints=False):
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    e_50, e_90 = np.median(qerror), np.percentile(qerror,90)    
    e_mean = np.mean(qerror)

    if prints:
        print("Median: {}".format(e_50))
        print("Mean: {}".format(e_mean))

    res = {
        'q_median' : e_50,
        'q_90' : e_90,
        'q_mean' : e_mean,
    }

    return res

def get_corr(ps, ls): # unnormalised
    ps = np.array(ps)
    ls = np.array(ls)
    corr, _ = pearsonr(np.log(ps), np.log(ls))
    
    return corr

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def eval_workload(database, workload, methods):

    get_table_sample = methods['get_sample']

    workload_file_name = './data/{}/workloads/'.format(database) + workload
    table_sample = get_table_sample(workload_file_name)
    plan_df = pd.read_csv('./data/{}/{}_plan.csv'.format(database, workload))
    workload_csv = pd.read_csv('./data/{}/workloads/{}.csv'.format(database, workload),sep='#',header=None)
    workload_csv.columns = ['table','join','predicate','cardinality']
    ds = PlanTreeDataset(plan_df, workload_csv, \
        methods['encoding'], methods['hist_file'], methods['cost_norm'], \
        methods['cost_norm'], 'cost', table_sample)

    eval_score = evaluate(methods['model'], ds, methods['bs'], methods['cost_norm'], methods['device'],True)
    return eval_score, ds


def select_pretrain_list(input_data, length):
    # split_size = 10
    split_size = int(length/2)
    # split_size = int(length/3)
    # split_size = int(length/4)
    q = [torch.quantile(input_data, i/split_size) for i in range(split_size+1)] 
    indices = [torch.nonzero((input_data >= q[i]) & (input_data < q[i+1]),as_tuple=True)[0] for i in range(split_size)]
    sampled_indices = [torch.randperm(len(indices[i]))[:length//split_size] for i in range(split_size)]
    sampled_indices = torch.cat([indices[i][sampled_indices[i]] for i in range(split_size)])
    # return random.sample(range(len(input_data)), length)
    return sorted(sampled_indices.tolist())

def pretraining_MoEPlan(model, pretrain_list, ds_all, df_test, args,\
    crit = nn.MSELoss(),crit_step1 = nn.CrossEntropyLoss(),optimizer=None, scheduler=None):
    
    to_pred, bs, device, epochs, clip_size, num_experts, cost_norm = \
        args.to_predict, args.bs, args.device, args.epochs, args.clip_size, args.num_experts, args.cost_norm
    lr = args.lr
    crit_pre = nn.BCEWithLogitsLoss()
    k = model.k
    a = 0.5
    b = 0.3
    # print("loss:loss0",a ,"loss1",1-a-b)
    rng = np.random.default_rng()
    train_idxs = np.array([i for i in range(len(ds_all))])
    epochs = 300 # pretrain epoch
    state_TS_once = True
    list_train = pretrain_list
    print("pretrain_list:",pretrain_list)
    for epoch in range(epochs):
        losses = 0
        cost_predss = np.empty(0)
        model.train()
        print("len0.8")
        train_idxs1 = np.array([i for i in list_train])
        moe_params = model.parameters()
        if not optimizer:
            optimizer = torch.optim.Adam(moe_params, lr=lr)
        for idxs in chunks(train_idxs1, bs):
            optimizer.zero_grad()
            batch, batch_labels = collator(list(zip(*[ds_all[pos1] for pos1 in idxs])))
            l = torch.stack([label[0] for label in batch_labels],dim = 0)
            batch = batch.to(device)
            gates,y_pred,y_min =model(batch)
            # gates,y_pred,y_min,sim_out =model(batch)
            y_pred = y_pred.squeeze()
            expert_index_min = torch.tensor([label[1] for label in batch_labels]).to(torch.long).to(device)
            l = l.to(device)
            l = l[:,:num_experts].to(torch.float)
            expert_index_min = torch.argmin(l,dim=1)
            l_min,_ = torch.min(l,dim=1)
            threshold = l_min*1.05
            label_g = (l < threshold[:, None]).float().softmax(dim=-1)
            #using
            if(model.is_sampler and state_TS_once):
                ra = torch.zeros(num_experts)
                rb = torch.zeros(num_experts)
                for i in range(l.shape[0]):
                    for j in range(num_experts):
                        if(j != expert_index_min[i]):
                            ra[j] += 1
                        else:
                            rb[j] += 1
            loss_1 = crit(y_pred, l)
            loss_2 = crit(y_min,l_min)
            loss_0 = crit_step1(gates, expert_index_min)
            loss_00 = crit_step1(gates, label_g)
            loss_000 = crit_pre(gates, label_g)
            # loss_01 = crit_pre(sim_out, label_g)
            # loss_001 = crit_step1(sim_out, expert_index_min)
            # loss = a * loss_0  + (1-a-b) * loss_1 + b * loss_2
            # loss = loss_0 + loss_1 + loss_2
            loss = loss_0 + loss_1 + loss_2
            # loss = a * loss_0  + (1-a-b) * loss_1 + b * loss_2 
            # loss = a * loss_00  + (1-a-b) * loss_1 +  b * loss_2
            # loss = loss_1 + loss_2 + b * loss_00
            # loss = loss_0 + loss_2
            # loss = loss_0 + loss_1 + loss_2 + loss_01
            # loss = loss_0 + loss_2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(moe_params, clip_size)
            optimizer.step()
            losses += loss.item()
        state_TS_once = False
        print("epoch:",epoch,",loss:",losses)
    model.pre_train = False
    return model

def training_MoEPlan(model, ds_all, df_test, args,\
    crit = nn.MSELoss(), crit_step1 = nn.CrossEntropyLoss(),optimizer=None, scheduler=None):
    to_pred, bs, device, epochs, clip_size, num_experts, lr, pretrain_len, cost_norm, k= \
        args.to_predict, args.bs, args.device, args.epochs, args.clip_size, args.num_experts, args.lr, args.pretrain_len, args.cost_norm, args.k
    crit_step2 = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng()
    train_idxs = np.array([i for i in range(len(ds_all))])
    if(model.pre_train):
        print("pretrain")
        # pretrain_list = select_pretrain_list(torch.Tensor([df_test.iloc[:,1][i][0] for i in range(int(0.5*len(df_test)))]).squeeze(),20)
        if(args.dataset == 'job'):
            # pretrain_list = [i for i in range(0,40,2)] #1b~20b
            # pretrain_select_data = torch.Tensor([df_test.iloc[:,1][i][0] for i in range(int(0.5*len(df_test)))]).squeeze()
            pretrain_select_data = torch.Tensor([df_test.iloc[:,1][i][0] for i in range(int(0.66*len(df_test)))]).squeeze()# valid5%
            # pretrain_select_data = torch.Tensor([df_test.iloc[:,1][i][0] for i in range(64)]).squeeze()# 1b~33b 1c~33c(no 24c 33c)
            # pretrain_select_data = torch.Tensor([df_test.iloc[:,1][i][0] for i in range(int(0.71*len(df_test)))]).squeeze()# 0.71*len(df_test) = 80
            pretrain_list = select_pretrain_list(pretrain_select_data,pretrain_len)
        elif(args.dataset == "tpcds_new"):
            # pretrain_select_data = torch.Tensor([df_test.iloc[:,1][i][0] for i in range(int(0.67*len(df_test)))]).squeeze()
            pretrain_select_data = torch.Tensor([df_test.iloc[:,1][i][0] for i in range(int(0.66*len(df_test)))]).squeeze()
            pretrain_list = select_pretrain_list(pretrain_select_data,pretrain_len)
        else:
            pretrain_select_data = torch.Tensor([df_test.iloc[:,1][i][0] for i in train_idxs[:int(0.8*len(train_idxs))]]).squeeze()
            pretrain_list = select_pretrain_list(pretrain_select_data,pretrain_len)
        model = pretraining_MoEPlan(model, pretrain_list ,ds_all, df_test, args, crit, crit_step1, optimizer, scheduler)
        model.pre_train = False
        # if(valid):
            # valid_list = select_valid_list(pretrain_select_data,valid_len,pretrain_list)
            # print("valid_list:",valid_list)
        print("end")
    # a= 1
    a = 0.5
    b = 0.3
    # a = 0.8
    # b = 0.1
    print("loss:loss0",a ,"loss1",1-a-b)
    use_df = torch.zeros(bs,num_experts).to(device)
    state_use_df = True
    valid_speedup = 0
    patience_count = 0
    model_best = None
    for epoch in range(epochs):
        losses = 0
        model.train()
        if(args.dataset == "job"):
            train_idxs1 = train_idxs[:int(0.71*len(train_idxs))] # 80 queries withou 1a~33a 
            # valid_idxs1 = train_idxs[int(0.66*len(train_idxs)):int(0.71*len(train_idxs))]
        elif(args.dataset == "tpcds"):
            train_idxs1 = train_idxs[:int(0.8*len(train_idxs))]
        elif(args.dataset == "tpcds_new"):
            train_idxs1 = train_idxs[:int(0.67*len(train_idxs))]
        else:
            print("other dataset")
            return
        
        moe_params = model.parameters()
        if not optimizer:
            optimizer = torch.optim.Adam(moe_params, lr=lr)
        if not scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.7)

        for idxs in chunks(train_idxs1, bs):
            optimizer.zero_grad()
            batch, batch_labels = collator(list(zip(*[ds_all[pos1] for pos1 in idxs])))
            l = torch.stack([label[0] for label in batch_labels],dim = 0)
            batch = batch.to(device)
            (cost_preds,cost_virsual), (select_index_pre,select_index), (gates_value,gates0), ts_select,gates_route = model(batch)
            cost_preds = cost_preds.squeeze()
            l = l.to(device)
            t_select = torch.gather(l,1,select_index_pre).to(torch.float)
            if(model.is_sampler):
                l_ts = torch.gather(l,1,ts_select.unsqueeze(1).to(args.device)).to(torch.float)
            expert_index_min = torch.argmin(t_select,dim=1)
            t_select_min,_ = torch.min(t_select,dim=1)
            threshold = t_select_min*1.05
            loss_2 = crit(cost_virsual, t_select_min)
            loss_1 = crit(cost_preds,t_select)
            loss_0 = crit_step1(gates_value, expert_index_min)
            # loss_0 = crit_step1(gates_value, label_g_p)
            # label_g = (t_select < threshold[:, None]).float()
            # label_g_p = (t_select < threshold[:, None]).float().softmax(dim=-1)
            # loss_000 = crit_step1(gates_route, label_g_p)
            # loss_00 = crit_step1(gates_route, expert_index_min)
            # loss_001 = crit_step2(gates_route, label_g)
            loss = a * loss_0  + (1-a-b) * loss_1 + b * loss_2 
            if(model.is_sampler):
                ra = torch.zeros(num_experts)
                rb = torch.zeros(num_experts)
                for i in range(len(ts_select)):
                    if(l[i][ts_select[i]] <= l[i][expert_index_min[i]]):
                        ra[ts_select[i]] += 1
                    else:
                        rb[ts_select[i]] += 1
                model.TSampler.update(ra,rb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(moe_params, clip_size)
            optimizer.step()
            losses += loss.item()
        print("epoch:",epoch,",loss:",losses)

        # test
        if(True):
            print("test!epoch:",epoch)
            # time_start = time.time()
            model.eval()
            if(args.dataset == "tpcds"):
                test_df = df_test.iloc[[j for j in train_idxs[int(0.8*len(train_idxs)):len(train_idxs)]],:]
                batch,batch_labels = collator(list(zip(*[ds_all[j] for j in train_idxs[int(0.8*len(train_idxs)):len(train_idxs)]])))            
            elif(args.dataset == "job"):
                # time_end1 = time.time()
                test_df = df_test.iloc[[j for j in range(int(0.71*len(ds_all)),len(ds_all))],:]
                # time_start1 = time.time()
                batch,batch_labels = collator(list(zip(*[ds_all[j] for j in range(int(0.71*len(ds_all)),len(ds_all))])))
            elif(args.dataset == "tpcds_new"):
                # time_end1 = time.time()
                test_df = df_test.iloc[[j for j in range(int(0.67*len(ds_all)),len(ds_all))],:]
                # time_start1 = time.time()
                batch,batch_labels = collator(list(zip(*[ds_all[j] for j in range(int(0.67*len(ds_all)),len(ds_all))])))
            else:
                print("wrong")
                return
            batch = batch.to(device)
            (cost_preds,cost_min), (select_index_pre,select_index), (gates_value,gates0), _ = model.inference(batch)
            # time_end2 = time.time()
            l = torch.stack([label[0] for label in batch_labels],dim = 0)
            l_min = l[:,num_experts]
            l_min = torch.FloatTensor(l_min.to(torch.float)).to(device)
            l_default = l[:,0]
            l_default = torch.FloatTensor(l_default.to(torch.float)).to(device)
            cost_preds = cost_preds.squeeze()
            expert_index_min = torch.tensor([label[1] for label in batch_labels]).to(torch.long).to(device)
            l = l.to(device)
            t_select = torch.gather(l,1,select_index_pre).to(torch.float)
            expert_index_min = torch.argmin(t_select,dim=1)
            l_expert = torch.gather(l,1,select_index).squeeze(1).to(torch.float)
            t_select_min,_ = torch.min(t_select,dim=1)
            loss_2 = crit(cost_min, t_select_min)
            loss_1 = crit(cost_preds,t_select)
            loss_0 = crit_step1(gates_value, expert_index_min)
            loss = a * loss_0  + (1-a-b) * loss_1 + b * loss_2
            expert_sum = 0
            select_index_df = select_index.detach().cpu().numpy()
            default_sum = 0
            min_sum = 0
            if(epoch % 100 == 99):
                print(select_index)
                # print((time_end2+time_end1-time_start-time_start1)*1000)
            expert_sum_each = [0 for index in range(args.num_experts)]
            gmrl_each = [1.0 for index in range(args.num_experts)]
            gmrl_ours = 1.0
            # if(epoch == epochs - 1):
            #     print(test_df[0])
            for u in range(len(test_df)):
                for index in range(args.num_experts):
                    expert_sum_each[index] += test_df.iloc[u,1][index]
                    gmrl_each[index] *= (test_df.iloc[u,1][index]/test_df.iloc[u,1][0])
                gmrl_ours *= (test_df.iloc[u,1][select_index_df[u][0]]/test_df.iloc[u,1][0])
                # if(epoch==99):
                    # print(select_index_df[u][0])
                    # print(test_df.iloc[u,1][select_index_df[u][0]])
                expert_sum += test_df.iloc[u,1][select_index_df[u][0]]
                default_sum += test_df.iloc[u,1][0]
                min_sum += test_df.iloc[u,1][num_experts]
                # if(epoch > 90 and test_df.iloc[k,1][select_index_df[k][0]] > test_df.iloc[k,1][num_experts]):
                #     print(test_df.iloc[k,1][select_index_df[k][0]]," ",test_df.iloc[k,1][num_experts],"\n")
                if(epoch>=epochs-1):
                    print(u,";",test_df.iloc[u,1][select_index_df[u][0]],";",test_df.iloc[u,1][0],";",test_df.iloc[u,1][num_experts])
            print("loss0:",loss_0)
            print("loss1:",loss_1)
            print("loss2:",loss_2)
            print("speed up:",default_sum/expert_sum)
            print("ours WRL:",expert_sum/default_sum)
            print("ours GMRL:",gmrl_ours**(1/len(test_df)))
            for index in range(args.num_experts):
                print("expert_",index,"speed_up:",default_sum/expert_sum_each[index])
                print("expert_",index,"GMRL:",gmrl_each[index] ** (1/len(test_df)))
            print(expert_sum,",",default_sum,",",min_sum)

        # if(patience_count > args.patience):
        #     print(epoch - args.patience)
        #     break
        scheduler.step()   
    return model

def logging(args, epoch, qscores, filename = None, save_model = False, model = None):
    arg_keys = [attr for attr in dir(args) if not attr.startswith('__')]
    arg_vals = [getattr(args, attr) for attr in arg_keys]
    
    res = dict(zip(arg_keys, arg_vals))
    model_checkpoint = str(hash(tuple(arg_vals))) + '.pt'

    res['epoch'] = epoch
    res['model'] = model_checkpoint 


    res = {**res, **qscores}

    filename = args.newpath + filename
    model_checkpoint = args.newpath + model_checkpoint
    
    if filename is not None:
        if os.path.isfile(filename):
            df = pd.read_csv(filename)
            df = df.append(res, ignore_index=True)
            df.to_csv(filename, index=False)
        else:
            df = pd.DataFrame(res, index=[0])
            df.to_csv(filename, index=False)
    if save_model:
        torch.save({
            'model': model.state_dict(),
            'args' : args
        }, model_checkpoint)
    
    return res['model']  
