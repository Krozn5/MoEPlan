import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
# import dgl
# from dgl.nn import GATConv
# from torch.nn import TransformerEncoder
# from torch.nn import TransformerDecoder
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos+x.size(0), :]
        return self.dropout(x)

class AnomalyTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.N, self.d_model, self.device = config.train.N, config.train.d_model, config.device

        self.attention = AnomalyAttention(config)
        self.ln1 = nn.LayerNorm(self.d_model)
        self.ff = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.ReLU())
        self.ln2 = nn.LayerNorm(self.d_model)

    def forward(self, x):
        x_identity = x
        x,S = self.attention(x)
        z = self.ln1(x + x_identity)

        z_identity = z
        z = self.ff(z)
        z = self.ln2(z + z_identity)

        return z,S

def scaled_dot_product_attention(q, k, v, scale=None, attn_mask=None):
    """
    Weighted sum of v according to dot product between q and k
    :param q: query tensor，[-1, L_q, V]
    :param k: key tensor，[-1, L_k, V]
    :param v: value tensor，[-1, L_k, V]
    :param scale: scalar
    :param attn_mask: [-1, L_q, L_k]
    """ 
    attention = torch.bmm(q, k.transpose(1, 2))  # [-1, L_q, L_k]
    if scale is not None:
        attention = attention * scale
	#    attention = attention - attention.max()
    if attn_mask is not None:
        attention = attention.masked_fill(attn_mask, -np.inf)
    attention = attention.softmax(dim=-1)
    context = torch.bmm(attention, v)  # [-1, L_q, V]
    return context

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, scale, attn_dropout=0.0):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, causality=True):

        attn = torch.matmul(q * self.scale, k.transpose(2, 3))

        if mask is not None:
            adder = (1 - mask).double() * -10000.0
            attn += adder
            #attn = attn.masked_fill(mask == 0, -1e9)
        if causality:
            diag_vals = torch.ones_like(attn[0, 0, :, :]) # (len, len)
            tril_vals = torch.tril(diag_vals) # (F, T)
            causal_masks = tril_vals.unsqueeze(0).unsqueeze(0).repeat(attn.shape[0], attn.shape[1], 1, 1)# (B, N, F, T)
   
            paddings = (1 - causal_masks).double() * -10000.0
            attn += paddings # (B, N, T_q, T_k)

        attn = self.dropout(F.softmax(attn, dim=-1))
        #attn = attention.softmax(dim=-1)
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)#, bias=True)
        self.w_ks = nn.Linear(d_model, n_head * d_k)#, bias=True)
        self.w_vs = nn.Linear(d_model, n_head * d_v)#, bias=True)
        #self.fc = nn.Linear(n_head * d_v, d_model)#, bias=True)

        self.attention = ScaledDotProductAttention(scale=d_k ** -0.5, attn_dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)#, eps=1e-8)


    def forward(self, q, k, v, mask=None, causality=True):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        print(q.size())
        print(self.w_qs(q).size())
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # logging.info("the shape for q is" + str(q.shape))
        # logging.info("the shape for k is" + str(k.shape))

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        print("q",q.size())
        q, attn = self.attention(q, k, v, mask=mask, causality=causality)
        print(q.size())
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        #q = self.dropout(self.fc(q))
        q += residual


        q = self.layer_norm(q)

        return q, attn

class PositionWiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None, causality=True):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask, causality=causality)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.0):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_inputs, enc_outputs, self_attn_mask=None, context_attn_mask=None):
        # self self_attn, all inputs are decoder inputs
        dec_output, self_attention = self.self_attn(
          dec_inputs, dec_inputs, dec_inputs, self_attn_mask)

        # context attention
        # query is decoder's outputs, key and value are encoder's inputs
        dec_output, context_attention = self.self_attn(
          enc_outputs, enc_outputs, dec_output, context_attn_mask)

        # decoder's output, or context
        dec_output = self.pos_ffn(dec_output)

        return dec_output, self_attention, context_attention

def padding_mask(seq_k, seq_q):
    # seq_k和seq_q的形状都是[B,L]
    len_q = seq_q.size(1)
    # `PAD` is 0
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask

class Encoder(nn.Module):
    """多层EncoderLayer组成Encoder。"""
    def __init__(self,config):
        super(Encoder, self).__init__()
        # vocab_size,
        # max_seq_len
        self.d_model = config.train.d_model
        self.n_head = config.train.n_head
        self.d_inner1 = config.train.d_inner1
        self.d = config.train.d
        head_emb_size = int(self.d_model / self.n_head)
        self.dropout=config.dropout
        self.mask = config.train.mask
        self.n_window = 10 #?


        self.emb = nn.Sequential(nn.Linear(self.d, self.d_model, bias=True), 
                nn.ReLU(True))

        self.pos_encoder = PositionalEncoding(self.d_model, 0.1, self.n_window)


        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(self.d_model,self.d_inner1,self.n_head,head_emb_size,head_emb_size) for _ in range(config.train.layers)]
        )

    def forward(self, inputs):
        output = self.emb(inputs)
        output = output.permute(1, 0, 2)
        output = self.pos_encoder(output)
        output = output.permute(1, 0, 2)
        if(self.mask):
            self_attention_mask = padding_mask(inputs, inputs)
        else:
            self_attention_mask = None

        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)

        return output, attentions

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.d_model = config.train.d_model
        self.d = config.train.d
        self.n_head = config.train.n_head
        self.d_inner2 = config.train.d_inner2
        head_emb_size = int(self.d_model / self.n_head)
        self.dropout=config.dropout
        self.mask = config.train.mask
        self.n_window = 10 #?

        self.emb = nn.Sequential(nn.Linear(self.d, self.d_model, bias=True), 
                nn.ReLU(True))

        self.pos_encoder = PositionalEncoding(self.d_model, 0.1, self.n_window)


        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(self.d_model,self.d_inner2,self.n_head,head_emb_size,head_emb_size) for _ in range(config.train.layers)]
        )


    def forward(self, inputs, enc_output, context_attn_mask=None):
        output = self.emb(inputs)
        output = output.permute(1, 0, 2)
        output = self.pos_encoder(output)
        output = output.permute(1, 0, 2)

        if(self.mask):
            self_attention_padding_mask = padding_mask(inputs, inputs)
            # self_attention_padding_mask = padding_mask(inputs, inputs)
            seq_mask = sequence_mask(inputs)
            self_attn_mask = torch.gt((self_attention_padding_mask + seq_mask), 0)

        else:
            self_attention_padding_mask = None
            self_attn_mask = None


        self_attentions = []
        context_attentions = []
        for decoder in self.decoder_layers:
            output, self_attn, context_attn = decoder(
            output, enc_output, self_attn_mask, context_attn_mask)
            self_attentions.append(self_attn)
            context_attentions.append(context_attn)
        return output, self_attentions, context_attentions


class Trans(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.N = config.train.N
        self.d_model = config.train.d_model
        self.d = config.train.d
        # self.emb = nn.Linear(self.d, self.d_model, bias=True)
        # self.d_model_2 = int(self.d_model / 2)
        # self.predict1 = nn.Linear(self.d_model, self.d_model_2, bias=True)
        # self.predict2 = nn.Linear(self.d_model_2, self.d, bias=True)
        self.n_window = 10 #?
        self.n_head = config.train.n_head
        self.lr = 0.002 #?0.0001
        self.batch = config.train.batch_size
        # self.d_k = self.d_model
        # self.d_v = self.d_model
        # # self.inner_times = config.train.inner_times
        # self.d_inner1 = config.train.d_inner1
        # self.d_inner2 = config.train.d_inner2
        self.name = "Trans"
        self.mask = config.train.mask
        head_emb_size = int(self.d_model / self.n_head)
        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.linear = nn.Linear(self.d_model, self.d, bias=False)
        # self.linear = nn.Linear(d_model, d, bias=True)
        self.softmax = nn.Softmax(dim=2)
        
        # self.output = None
        # self.lambda_ = config.train.lambda_

        # self.P_layers = []
        # self.S_layers = []

        self.device = config.device

    def forward(self, src, tgt):
        if(self.mask):
            context_attn_mask = padding_mask(tgt, src)
        else:
            context_attn_mask = None

        output, enc_self_attn = self.encoder(src)
        output, dec_self_attn, ctx_attn = self.decoder(tgt, output, context_attn_mask)
        output = self.linear(output)
        gamma = self.softmax(output)

        return output, enc_self_attn, dec_self_attn, ctx_attn, gamma

torch.manual_seed(1)