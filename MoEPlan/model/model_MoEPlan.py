import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
from .model import QueryFormer,FeedForwardNetwork
from .util import clone_module
from .transformer import MultiHeadAttention

class MoEModel(nn.Module):
    def __init__(self, num_experts, expert_size, output_size, k,emb_size = 64 ,ffn_dim = 32, head_size = 8, \
                dropout = 0.1, attention_dropout_rate = 0.1, n_layers = 8, \
                use_sample = True, use_hist = True, bin_number = 50, \
                pred_hid = 256,QFmodel = None, k1 = 9,device='cuda'):
        # *** expert_size 在sim下改成没用的参数 默认为hidden_dim
        super(MoEModel, self).__init__()
        # if(self.gate_method != "sim_emb" and self.gate_method != "sim_emb_twice"):
            # k = 1
        if use_hist:
            hidden_dim = emb_size * 5 + emb_size //8 + 1
        else: 
            hidden_dim = emb_size * 4 + emb_size //8 + 1
        self.expert_size = hidden_dim
        self.embedding_dim = hidden_dim
        self.emb_expert = nn.Embedding(num_embeddings= num_experts,embedding_dim= self.embedding_dim)
        self.device = device
        self.emb_size = emb_size
        self.hidden_dim = hidden_dim
        self.head_size = head_size
        self.use_sample = use_sample
        self.use_hist = use_hist
        if(QFmodel != None):
            self.emb_layer = QFmodel
        else: 
            self.emb_layer = QueryFormer(emb_size = emb_size ,ffn_dim = ffn_dim, head_size = head_size, dropout = dropout, n_layers = n_layers, use_sample = True, use_hist = True, pred_hid = pred_hid)
        self.TSampler = ThompsonSampling(num_experts,device)
        self.num_experts = num_experts
        self.k = k
        self.softmax = nn.Softmax(dim=-1)
        self.out_mlp = nn.Linear(self.expert_size, 1)
        self.gate_weights = nn.Parameter(torch.Tensor(self.num_experts, self.hidden_dim))
        self.expert_weights = nn.Parameter(torch.Tensor(self.num_experts, self.hidden_dim, self.expert_size))
        self.expert_min = nn.Parameter(torch.Tensor(self.hidden_dim, self.expert_size))
        self.softmax = nn.Softmax()
        self.is_sampler = True
        self.pre_train = True
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.expert_weights, 0, 0.01)
        nn.init.normal_(self.expert_min, 0, 0.01)
        nn.init.normal_(self.gate_weights, 0, 0.01)

    def pick_Ts(self,index,probs):
        new_index = torch.zeros((index.shape[0], index.shape[1]+1), dtype=torch.int64,device = self.device)
        new_index[:,:index.shape[1]]=index
        for i in range(index.shape[0]):
            random_p = self.TSampler.sample_p(probs[i])
            _,pick_index = torch.topk(random_p,index.shape[1]+1)
            for j in range(index.shape[1]+1):
                if(pick_index[j] not in index[i]):
                    new_index[i,index.shape[1]] = pick_index[j]
                    break
        return new_index

    def pick_Ts_infer(self,index,probs):
        new_index = torch.zeros((index.shape[0], index.shape[1]+1), dtype=torch.int64,device = self.device)
        new_index[:,:index.shape[1]]=index
        for i in range(index.shape[0]):
            random_p = self.TSampler.inference_p(probs[i])
            _,pick_index = torch.topk(random_p,index.shape[1]+1)
            for j in range(index.shape[1]+1):
                if(pick_index[j] not in index[i]):
                    new_index[i,index.shape[1]] = pick_index[j]
                    break
        return new_index

    def inference(self,x):
        x = self.emb_layer(x)
        x = torch.squeeze(x[:,0,:],dim=1)
        expert_min_stdvec = torch.matmul(x,self.expert_min)
        expert_min_out = self.out_mlp(expert_min_stdvec)
        gates = F.linear(x, self.gate_weights)
        if(self.pre_train):
            experts_embedding = torch.cat([torch.matmul(x,self.expert_weights[i]).unsqueeze(1) \
                                       for i in range(self.num_experts)],dim=1)
            y_pred = self.out_mlp(experts_embedding)
            return gates,y_pred,expert_min_out   
        if(self.is_sampler == 1):
            top_k_gates, top_k_indices = torch.topk(gates, k=self.k - 1, dim=-1)
            gates_tmp = torch.zeros_like(gates, requires_grad=True).scatter_add(1, top_k_indices, top_k_gates)
            top_k_indices = self.pick_Ts_infer(top_k_indices, gates)
            select_probability = gates_tmp
        else:
            top_k_logits, top_k_indices = torch.topk(gates, k=self.k, dim=-1)
            top_k_gates = self.softmax(top_k_logits)
            select_probability = torch.zeros_like(gates, requires_grad=True).scatter_add(1, top_k_indices, top_k_gates)
        experts_embedding_select = torch.cat([torch.matmul(x[i], \
            torch.index_select(self.expert_weights,0,top_k_indices[i])).unsqueeze(0) \
            for i in range(x.shape[0])],dim = 0)
        out_select = self.out_mlp(experts_embedding_select)
        experts_embedding_select = experts_embedding_select.transpose(0,1)
        sim_out = torch.cat([F.cosine_similarity(expert_min_stdvec, \
                    experts_embedding_select[i]).unsqueeze(1) for i in range(self.k)],dim=1)
        prob_out_select = sim_out
        top_k_sim_prob, top_k_sim_indices = torch.topk(prob_out_select, 1, dim=-1)
        top_k_sim_prob = self.softmax(top_k_sim_prob)
        select_indices = torch.gather(top_k_indices,1,top_k_sim_indices)
        # return (out_select,expert_min_out), (top_k_indices,select_indices), (sim_out,(gates,y_view)) , top_k_indices[:,-1]
        return (out_select,expert_min_out), (top_k_indices,select_indices), (sim_out,select_probability) , top_k_indices[:,-1]

    def print_ab(self):
        print("a:",self.TSampler._a)
        print("b:",self.TSampler._b)
    
    def forward(self, x):
        x = self.emb_layer(x)
        x = torch.squeeze(x[:,0,:],dim=1)
        expert_min_stdvec = torch.matmul(x,self.expert_min)
        expert_min_out = self.out_mlp(expert_min_stdvec)
        gates = F.linear(x, self.gate_weights)
        if(self.pre_train):
            experts_embedding = torch.cat([torch.matmul(x,self.expert_weights[i]).unsqueeze(1) \
                                       for i in range(self.num_experts)],dim=1)
            y_pred = self.out_mlp(experts_embedding)
            return gates,y_pred,expert_min_out
        if(self.is_sampler == 1):
            top_k_gates, top_k_indices = torch.topk(gates, k=self.k - 1, dim=-1)
            top_k_gates_return,_ = torch.topk(gates, k=self.k, dim=-1)
            gates_tmp = torch.zeros_like(gates, requires_grad=True).scatter_add(1, top_k_indices, top_k_gates)
            top_k_indices = self.pick_Ts(top_k_indices, gates)
            select_probability = gates_tmp
        else:
            top_k_logits, top_k_indices = torch.topk(gates, k=self.k, dim=-1)
            top_k_gates_return = top_k_logits
            top_k_gates = self.softmax(top_k_logits)
            select_probability = torch.zeros_like(gates, requires_grad=True).scatter_add(1, top_k_indices, top_k_gates)
        experts_embedding_select = torch.cat([torch.matmul(x[i], \
            torch.index_select(self.expert_weights,0,top_k_indices[i])).unsqueeze(0) \
            for i in range(x.shape[0])],dim = 0)
        experts_embedding = torch.cat([torch.matmul(x,self.expert_weights[i]).unsqueeze(1) \
                                       for i in range(self.num_experts)],dim=1)
        y_view = self.out_mlp(experts_embedding)
        out_select = self.out_mlp(experts_embedding_select)
        experts_embedding_select = experts_embedding_select.transpose(0,1)
        sim_out = torch.cat([F.cosine_similarity(expert_min_stdvec, \
                    experts_embedding_select[i]).unsqueeze(1) for i in range(self.k)],dim=1)
        # prob_out_select = self.p * sim_out + (1-self.p) * (out_select * -1).squeeze()
        prob_out_select = sim_out
        # sim_out = prob_out_select
        top_k_sim_prob, top_k_sim_indices = torch.topk(prob_out_select, 1, dim=-1)
        top_k_sim_prob = self.softmax(top_k_sim_prob)
        select_indices = torch.gather(top_k_indices,1,top_k_sim_indices)
        return (out_select,expert_min_out), (top_k_indices,select_indices), (sim_out,(gates,y_view)) , top_k_indices[:,-1], top_k_gates_return
        # return (out_select,expert_min_out), (top_k_indices,select_indices), (sim_out,select_probability) , top_k_indices[:,-1]

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

class ThompsonSampling:
    def __init__(self, K, device):
        self._a = torch.ones(K)
        self._b = torch.ones(K)
        self.device = device
        self.a = 0.1
    
    def print_ab(self):
        print("a:",self._a)
        print("b:",self._b)
    
    def sample_p(self,probs):
        # 1
        beta_dist = dist.Beta(self._a, self._b)
        samples1 = beta_dist.sample().to(self.device)
        samples2 = probs
        samples = self.a * samples1 + (1 - self.a) * samples2
        return samples

    def inference_p(self,probs):
        samples = self._a / (self._a + self._b)
        samples = samples.to(self.device)
        samples = self.a * samples + (1 - self.a) * probs
        return samples

    def update(self,ra,rb):
        self._a += ra
        self._b += rb
        return