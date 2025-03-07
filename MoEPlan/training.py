import numpy as np
import os
import torch
import torch.nn as nn
import time
import pandas as pd
from scipy.stats import pearsonr
from model.database_util import get_hist_file, get_job_table_sample, collator
from model.model import QueryFormer
from model.trainer import training_MoEPlan
from model.dataset_util import get_dataset
from model.util import Normalizer
 

class Args:
    bs = 32
    lr = 0.0001 
    epochs = 100
    clip_size = 50
    embed_size = 64
    emb_size = 64
    pred_hid = 128
    ffn_dim = 128
    head_size = 12
    n_layers = 8
    dropout = 0.1
    sch_decay = 0.6
    num_experts = 16
    a = 0.5
    b = 0.3
    device = 'cuda:0'
    newpath = './results/full/cost/'
    to_predict = 'cost'
    dataset = 'job' # 'job' or 'tpcds' or 'tpcds_new'
    pretrain_len = 20
    k=3 # select
    is_sampler=1 # true
    cost_norm = Normalizer(0, 10)
    card_norm = Normalizer(1, 100)
args = Args()
import os
if not os.path.exists(args.newpath):
    os.makedirs(args.newpath)
from model.util import seed_everything
seed_everything()
df, ds = get_dataset(args=args)
from model.model_MoEPlan import MoEModel
model = MoEModel(num_experts = args.num_experts, expert_size = args.emb_size, output_size = 1, k=args.k, \
                    emb_size = args.embed_size ,ffn_dim = args.ffn_dim, head_size = args.head_size, \
                    dropout = args.dropout, n_layers = args.n_layers, use_sample = True, use_hist = True, \
                    pred_hid = args.pred_hid, QFmodel = None,device=args.device
                )
_ = model.to(args.device)
model.is_sampler = args.is_sampler
model = training_MoEPlan(model, ds, df, args)
