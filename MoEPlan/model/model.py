import torch
from torch.utils.data import Dataset
import json
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

class Prediction(nn.Module):
    def __init__(self, in_feature = 69, hid_units = 256, contract = 1, mid_layers = True, res_con = True):
        super(Prediction, self).__init__()
        self.mid_layers = mid_layers
        self.res_con = res_con
        
        self.out_mlp1 = nn.Linear(in_feature, hid_units)

        self.mid_mlp1 = nn.Linear(hid_units, hid_units//contract)
        self.mid_mlp2 = nn.Linear(hid_units//contract, hid_units)

        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, features):
        
        hid = F.relu(self.out_mlp1(features))
        if self.mid_layers:
            mid = F.relu(self.mid_mlp1(hid))
            mid = F.relu(self.mid_mlp2(mid))
            if self.res_con:
                hid = hid + mid
            else:
                hid = mid
        out = torch.sigmoid(self.out_mlp2(hid))

        return out

        
class FeatureEmbed(nn.Module):
    # def __init__(self, embed_size=32, tables = 10, types=20, joins = 40, columns= 30, \
    #              ops=4, use_sample = True, use_hist = True, bin_number = 50):
    def __init__(self, embed_size=32, tables = 30, types=25, joins = 260, columns= 450, \
                 ops=10, use_sample = True, use_hist = True, bin_number = 50):
        super(FeatureEmbed, self).__init__()
        
        self.use_sample = use_sample
        self.embed_size = embed_size        
        
        self.use_hist = use_hist
        self.bin_number = bin_number
        
        self.typeEmbed = nn.Embedding(types, embed_size)
        self.tableEmbed = nn.Embedding(tables, embed_size)
        
        self.columnEmbed = nn.Embedding(columns, embed_size)
        self.opEmbed = nn.Embedding(ops, embed_size//8)

        self.linearFilter2 = nn.Linear(embed_size+embed_size//8+1, embed_size+embed_size//8+1)
        self.linearFilter = nn.Linear(embed_size+embed_size//8+1, embed_size+embed_size//8+1)

        self.linearType = nn.Linear(embed_size, embed_size)
        
        self.linearJoin = nn.Linear(embed_size, embed_size)
        
        self.linearSample = nn.Linear(1000, embed_size)
        
        self.linearHist = nn.Linear(bin_number, embed_size)

        self.joinEmbed = nn.Embedding(joins, embed_size)
        
        if use_hist:
            self.project = nn.Linear(embed_size*5 + embed_size//8+1, embed_size*5 + embed_size//8+1)
        else:
            self.project = nn.Linear(embed_size*4 + embed_size//8+1, embed_size*4 + embed_size//8+1)
    
    # input: B by 14 (type, join, f1, f2, f3, mask1, mask2, mask3)
    def forward(self, feature):

        typeId, joinId, filtersId, filtersMask, hists, table_sample = torch.split(feature,(1,1,9,3,self.bin_number*3,1001), dim = -1)
        
        typeEmb = self.getType(typeId)
        joinEmb = self.getJoin(joinId)
        filterEmbed = self.getFilter(filtersId, filtersMask)
        
        histEmb = self.getHist(hists, filtersMask)
        tableEmb = self.getTable(table_sample)
    
        if self.use_hist:
            final = torch.cat((typeEmb, filterEmbed, joinEmb, tableEmb, histEmb), dim = 1)
        else:
            final = torch.cat((typeEmb, filterEmbed, joinEmb, tableEmb), dim = 1)
        final = F.leaky_relu(self.project(final))
        
        return final
    
    def getType(self, typeId):
        emb = self.typeEmbed(typeId.long())

        return emb.squeeze(1)
    
    def getTable(self, table_sample):
        table, sample = torch.split(table_sample,(1,1000), dim = -1)
        emb = self.tableEmbed(table.long()).squeeze(1)
        
        if self.use_sample:
            emb += self.linearSample(sample)
        return emb
    
    def getJoin(self, joinId):
        emb = self.joinEmbed(joinId.long())

        return emb.squeeze(1)

    def getHist(self, hists, filtersMask):
        # batch * 50 * 3
        histExpand = hists.view(-1,self.bin_number,3).transpose(1,2)
        
        emb = self.linearHist(histExpand)
        emb[~filtersMask.bool()] = 0.  # mask out space holder
        
        ## avg by # of filters
        num_filters = torch.sum(filtersMask,dim = 1)
        total = torch.sum(emb, dim = 1)
        avg = total / num_filters.view(-1,1)
        avg = torch.where(torch.isnan(avg), torch.full_like(avg, 0), avg)
        
        return avg
        
    def getFilter(self, filtersId, filtersMask):
        ## get Filters, then apply mask
        filterExpand = filtersId.view(-1,3,3).transpose(1,2)
        colsId = filterExpand[:,:,0].long()
        opsId = filterExpand[:,:,1].long()
        vals = filterExpand[:,:,2].unsqueeze(-1) # b by 3 by 1
        
        # b by 3 by embed_dim
        
        col = self.columnEmbed(colsId)
        op = self.opEmbed(opsId)
        
        concat = torch.cat((col, op, vals), dim = -1)
        concat = F.leaky_relu(self.linearFilter(concat))
        concat = F.leaky_relu(self.linearFilter2(concat))
        
        ## apply mask
        concat[~filtersMask.bool()] = 0.
        
        ## avg by # of filters
        num_filters = torch.sum(filtersMask,dim = 1)
        total = torch.sum(concat, dim = 1)
        avg = total / num_filters.view(-1,1)
        avg = torch.where(torch.isnan(avg), torch.full_like(avg, 0), avg)
                
        return avg
    
#     def get_output_size(self):
#         size = self.embed_size * 5 + self.embed_size // 8 + 1
#         return size



class QueryFormer(nn.Module):
    def __init__(self, emb_size = 32 ,ffn_dim = 32, head_size = 8, \
                 dropout = 0.1, attention_dropout_rate = 0.1, n_layers = 8, \
                 use_sample = True, use_hist = True, bin_number = 50, \
                 pred_hid = 256
                ):
        
        super(QueryFormer,self).__init__()
        if use_hist:
            hidden_dim = emb_size * 5 + emb_size //8 + 1
        else:
            hidden_dim = emb_size * 4 + emb_size //8 + 1
        self.hidden_dim = hidden_dim
        self.head_size = head_size
        self.use_sample = use_sample
        self.use_hist = use_hist

        self.rel_pos_encoder = nn.Embedding(64, head_size, padding_idx=0)

        self.height_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
        
        self.input_dropout = nn.Dropout(dropout)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout, attention_dropout_rate, head_size)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        
        self.final_ln = nn.LayerNorm(hidden_dim)
        
        self.super_token = nn.Embedding(1, hidden_dim)
        self.super_token_virtual_distance = nn.Embedding(1, head_size)
        
        
        self.embbed_layer = FeatureEmbed(emb_size, use_sample = use_sample, use_hist = use_hist, bin_number = bin_number)
        
        self.pred = Prediction(hidden_dim, pred_hid)

        # if multi-task
        self.pred2 = Prediction(hidden_dim, pred_hid)
        self.init_weights()
    
    def init_weights(self):
        # Initialize embeddings with normal distribution
        nn.init.normal_(self.rel_pos_encoder.weight, std=0.02)
        nn.init.normal_(self.height_encoder.weight, std=0.02)
        nn.init.normal_(self.super_token.weight, std=0.02)
        nn.init.normal_(self.super_token_virtual_distance.weight, std=0.02)

        # Initialize layers with Xavier initialization
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    nn.init.normal_(param,std=0.02)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)

        # Initialize prediction layers with Xavier initialization
        for name, param in self.pred.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param,std=0.02)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

        if hasattr(self, 'pred2'):
            for name, param in self.pred2.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)

    def forward(self, batched_data):
        attn_bias, rel_pos, x = batched_data.attn_bias, batched_data.rel_pos, batched_data.x
        heights = batched_data.heights     
        
        n_batch, n_node = x.size()[:2]
        tree_attn_bias = attn_bias.clone()
        tree_attn_bias = tree_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1) 
        
        # rel pos
        rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(0, 3, 1, 2) # [n_batch, n_node, n_node, n_head] -> [n_batch, n_head, n_node, n_node]
        tree_attn_bias[:, :, 1:, 1:] = tree_attn_bias[:, :, 1:, 1:] + rel_pos_bias


        # reset rel pos here
        t = self.super_token_virtual_distance.weight.view(1, self.head_size, 1)
        tree_attn_bias[:, :, 1:, 0] = tree_attn_bias[:, :, 1:, 0] + t
        tree_attn_bias[:, :, 0, :] = tree_attn_bias[:, :, 0, :] + t
        
        x_view = x.view(-1, 1165)
        node_feature = self.embbed_layer(x_view).view(n_batch,-1, self.hidden_dim)
        
        # -1 is number of dummy
        
        node_feature = node_feature + self.height_encoder(heights)
        super_token_feature = self.super_token.weight.unsqueeze(0).repeat(n_batch, 1, 1)
        super_node_feature = torch.cat([super_token_feature, node_feature], dim=1)        
        
        # transfomrer encoder
        output = self.input_dropout(super_node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, tree_attn_bias)
        output = self.final_ln(output)
        return output
        output = torch.squeeze(output[:,0,:],dim=1) 
        # print(output.size())
        # print(output.size())
        # return
        return output
        # return self.pred(output[:,0,:]), self.pred2(output[:,0,:])



class MultiQueryFormer(nn.Module):
    def __init__(self, emb_size = 32 ,ffn_dim = 32, head_size = 8, \
                 dropout = 0.1, attention_dropout_rate = 0.1, n_layers = 8, \
                 use_sample = True, use_hist = True, bin_number = 50, \
                 pred_hid = 256, query_num = 8
                ):
        
        super(MultiQueryFormer,self).__init__()
        if use_hist:
            hidden_dim = emb_size * 5 + emb_size //8 + 1
        else:
            hidden_dim = emb_size * 4 + emb_size //8 + 1
        self.hidden_dim = hidden_dim
        self.head_size = head_size
        self.use_sample = use_sample
        self.use_hist = use_hist
        self.query_num = query_num

        self.rel_pos_encoder = nn.Embedding(64, head_size, padding_idx=0)

        self.height_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
        
        self.input_dropout = nn.Dropout(dropout)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout, attention_dropout_rate, head_size)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        
        self.final_ln = nn.LayerNorm(hidden_dim)
        
        self.super_token = nn.Embedding(1, hidden_dim)
        self.super_token_virtual_distance = nn.Embedding(1, head_size)
        
        
        self.embbed_layer = FeatureEmbed(emb_size, use_sample = use_sample, use_hist = use_hist, bin_number = bin_number)
        
        self.pred = Prediction(hidden_dim * query_num, pred_hid)

        # if multi-task
        self.pred2 = Prediction(hidden_dim * query_num, pred_hid)
        
    def forward(self, batched_data):
        attn_bias, rel_pos, x = batched_data.attn_bias, batched_data.rel_pos, batched_data.x
        heights = batched_data.heights     
        
        n_batch, n_node = x.size()[:2]
        tree_attn_bias = attn_bias.clone()
        tree_attn_bias = tree_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1) 
        
        # rel pos
        rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(0, 3, 1, 2) # [n_batch, n_node, n_node, n_head] -> [n_batch, n_head, n_node, n_node]
        tree_attn_bias[:, :, 1:, 1:] = tree_attn_bias[:, :, 1:, 1:] + rel_pos_bias


        # reset rel pos here
        t = self.super_token_virtual_distance.weight.view(1, self.head_size, 1)
        tree_attn_bias[:, :, 1:, 0] = tree_attn_bias[:, :, 1:, 0] + t
        tree_attn_bias[:, :, 0, :] = tree_attn_bias[:, :, 0, :] + t
        
        x_view = x.view(-1, 1165)
        node_feature = self.embbed_layer(x_view).view(n_batch,-1, self.hidden_dim)
        
        # -1 is number of dummy
        
        node_feature = node_feature + self.height_encoder(heights)
        super_token_feature = self.super_token.weight.unsqueeze(0).repeat(n_batch, 1, 1)
        super_node_feature = torch.cat([super_token_feature, node_feature], dim=1)        
        
        # transfomrer encoder
        output = self.input_dropout(super_node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, tree_attn_bias)
        output = self.final_ln(output)

        # aggregate multi-query embedding
        if output.shape[0] % self.query_num != 0:
            raise Exception('shape of batched_data inconsistent with query_num')
        output = output[:,0,:]
        output = torch.cat([torch.cat([output[i+j] for j in range(self.query_num)], dim=0).unsqueeze(0) for i in range(0, output.shape[0], self.query_num)], dim=0)


        
        return self.pred(output), self.pred2(output)


class QueryFormerMoE(nn.Module):
    def __init__(self, emb_size = 32 ,ffn_dim = 32, head_size = 8, \
                 dropout = 0.1, attention_dropout_rate = 0.1, n_layers = 8, \
                 use_sample = True, use_hist = True, bin_number = 50, \
                 pred_hid = 256, num_experts = 16,select_expert = 3
                ):
        
        super(QueryFormerMoE,self).__init__()
        if use_hist:
            hidden_dim = emb_size * 5 + emb_size //8 + 1
        else:
            hidden_dim = emb_size * 4 + emb_size //8 + 1
        self.hidden_dim = hidden_dim
        self.head_size = head_size
        self.use_sample = use_sample
        self.use_hist = use_hist
        self.num_experts = num_experts
        self.rel_pos_encoder = nn.Embedding(64, head_size, padding_idx=0)

        self.height_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
        
        self.input_dropout = nn.Dropout(dropout)
        # self.layers = nn.ModuleList([EncoderLayer(hidden_dim, ffn_dim, dropout, attention_dropout_rate, head_size)
        #             for _ in range(n_layers)])
        self.begin_ln = nn.LayerNorm(hidden_dim)
        self.final_ln = nn.LayerNorm(hidden_dim)
        
        self.super_token = nn.Embedding(1, hidden_dim)
        self.super_token_virtual_distance = nn.Embedding(1, head_size)
        self.embbed_layer = FeatureEmbed(emb_size, use_sample = use_sample, use_hist = use_hist, bin_number = bin_number)
        # self.pred = Prediction(hidden_dim, pred_hid)
        self.pred = nn.Linear(hidden_dim,1)
        # if multi-task
        # self.pred2 = Prediction(hidden_dim, pred_hid)
        self.gate_weights = nn.Parameter(torch.Tensor(self.num_experts, self.hidden_dim))
        self.expert_layers = nn.ModuleList(
                                [nn.ModuleList(
                                    [EncoderLayer(hidden_dim, ffn_dim, dropout, attention_dropout_rate, head_size)
                                for _ in range(n_layers)])
                            for _ in range(num_experts)])
        self.trans_layers = nn.ModuleList(
                                    [EncoderLayer(hidden_dim, ffn_dim, dropout, attention_dropout_rate, head_size)
                                for _ in range(n_layers)])
        # self.out_layers = Prediction(hidden_dim, pred_hid)
        self.num_experts = num_experts
        self.select_expert = select_expert 
        self.softmax = nn.Softmax(dim=-1)
        self.gate_method = "sim_emb"
        self.train_step = 1
        self.init_weights()
    
    def init_weights(self):
        # Initialize embeddings with normal distribution
        nn.init.normal_(self.rel_pos_encoder.weight, std=0.02)
        nn.init.normal_(self.height_encoder.weight, std=0.02)
        nn.init.normal_(self.super_token.weight, std=0.02)
        nn.init.normal_(self.super_token_virtual_distance.weight, std=0.02)
        nn.init.normal_(self.gate_weights, 0, 0.01)
        # Initialize layers with Xavier initialization
        for layers in self.expert_layers:
            for layer in layers:
                for name, param in layer.named_parameters():
                    if 'weight' in name:
                        nn.init.normal_(param,std=0.02)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)

        # Initialize layers with Xavier initialization
        for layer in self.trans_layers:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    nn.init.normal_(param,std=0.02)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)


        # Initialize prediction layers with Xavier initialization
        for name, param in self.pred.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param,std=0.02)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        

        # if hasattr(self, 'pred2'):
        #     for name, param in self.pred2.named_parameters():
        #         if 'weight' in name:
        #             nn.init.xavier_uniform_(param)
        #         elif 'bias' in name:
        #             nn.init.constant_(param, 0.0)

    def enc_i(self,x,i):
        for enc_layer in self.expert_layers[i]:
            x = enc_layer(x)
        return x 

    def enc_all(self,x):
        return torch.cat([self.enc_i(x,i).unsqueeze(0) for i in range(self.num_experts)],dim=0)
    
    def encode(self, batched_data):
        attn_bias, rel_pos, x = batched_data.attn_bias, batched_data.rel_pos, batched_data.x
        heights = batched_data.heights     
        n_batch, n_node = x.size()[:2]
        tree_attn_bias = attn_bias.clone()
        tree_attn_bias = tree_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1) 
        
        # rel pos
        rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(0, 3, 1, 2) # [n_batch, n_node, n_node, n_head] -> [n_batch, n_head, n_node, n_node]
        tree_attn_bias[:, :, 1:, 1:] = tree_attn_bias[:, :, 1:, 1:] + rel_pos_bias


        # reset rel pos here
        t = self.super_token_virtual_distance.weight.view(1, self.head_size, 1)
        tree_attn_bias[:, :, 1:, 0] = tree_attn_bias[:, :, 1:, 0] + t
        tree_attn_bias[:, :, 0, :] = tree_attn_bias[:, :, 0, :] + t
        
        x_view = x.view(-1, 1165)
        node_feature = self.embbed_layer(x_view).view(n_batch,-1, self.hidden_dim)
        
        # -1 is number of dummy
        
        node_feature = node_feature + self.height_encoder(heights)
        super_token_feature = self.super_token.weight.unsqueeze(0).repeat(n_batch, 1, 1)
        super_node_feature = torch.cat([super_token_feature, node_feature], dim=1)        
        
        # transfomrer encoder & MoE

        output = self.input_dropout(super_node_feature)
        for enc_layer in self.trans_layers:
            output = enc_layer(output, tree_attn_bias)
        output = self.begin_ln(output)
        return output

    def forward(self, batched_data):
        
        output = self.encode(batched_data)
        gates = F.linear(torch.squeeze(output[:,0,:],dim=1), self.gate_weights)
        # print(gates.size())
        top_k_gates, top_k_indices = torch.topk(gates, k=self.select_expert, dim=-1)
        experts_embedding_all = self.enc_all(output)[:,:,0,:]
        # print(experts_embedding_all.size())
        experts_embedding_select = torch.gather(experts_embedding_all,0,top_k_indices.transpose(0,1).unsqueeze(-1).repeat(1,1,output.size()[2]))
        experts_embedding_select = experts_embedding_select.transpose(0,1)
        # output = output.reshape(output.size()[0],output.size()[1])
        # out_select = self.out_mlp(experts_embedding_select)
        # experts_embedding_select = experts_embedding_select.squeeze(2)
        out_select = self.pred(experts_embedding_select) # (batch-size,self.select_expert,1)
        experts_embedding_select = experts_embedding_select.transpose(0,1)
        
        sim_out = torch.cat([F.cosine_similarity(output[:,0,:], experts_embedding_select[i]).unsqueeze(1) for i in range(self.select_expert)],dim=1)
        out = sim_out
        # print(out.size())
        # out = out_select.squeeze(-1)
        # print(out.size())
        # return
        top_k_sim_prob, top_k_sim_indices = torch.topk(out, 1, dim=-1)
        top_k_sim_prob = self.softmax(top_k_sim_prob)
        select_indices = torch.gather(top_k_indices,1,top_k_sim_indices)
        return out_select, (top_k_indices,select_indices), out
        
        for enc_layer in self.layers:
            output = enc_layer(output, tree_attn_bias)
        output = self.final_ln(output)
        return output
        output = torch.squeeze(output[:,0,:],dim=1) 
        # print(output.size())
        # print(output.size())
        # return
        return output
        # return self.pred(output[:,0,:]), self.pred2(output[:,0,:])



class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_size):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x























