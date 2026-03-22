import torch
import torch.nn as nn
import numpy as np
import math

def ScaledDotProductAttention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    
    scores = Q @ K.transpose(-2, -1)  # (batch, seq_len, seq_len)
    scores = scores / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    scores = torch.softmax(scores, dim=-1)
    output = scores @ V
    
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch = Q.shape[0]
        q_seq_len = Q.shape[1]
        k_seq_len = K.shape[1]
        
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        Q = Q.view(batch, q_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch, k_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch, k_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        output = ScaledDotProductAttention(Q, K, V, mask)
        
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, q_seq_len, self.num_heads * self.d_k)
        
        output = self.W_o(output)
        
        return output
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        # create a matrix of shape (max_seq_len, d_model)
        self.matrix = torch.zeros(max_seq_len,d_model)
        # fill even dims with sin, odd dims with cos

        positions = torch.arange(0, max_seq_len).unsqueeze(1)  
        dims = torch.arange(0, d_model, 2)                     
        denominator = torch.pow(10000, dims / d_model) 

        self.matrix[:,0::2] = torch.sin(positions/denominator) 
        self.matrix[:,1::2] = torch.cos(positions/denominator) 

        self.register_buffer('pe', self.matrix)      

        
    def forward(self, x):
        # add positional encoding to x and return
        seq_len = x.shape[1]
        x = x + self.pe[:seq_len,:]

        return x   
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # TODO: two linear layers
        self.l1 = nn.Linear(d_model,d_ff)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(d_ff,d_model)
        
    def forward(self, x):
        # TODO: linear → relu → linear
        
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)

        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # you need:
        # - MultiHeadAttention
        self.mha = MultiHeadAttention(d_model,num_heads)
        # - FeedForward
        self.ffn = FeedForward(d_model,d_ff)
        # - two LayerNorms → nn.LayerNorm(d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        # - dropout → nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # attention → add & norm
        x = self.layer_norm1(x + self.mha(x, x, x, mask))
    
        # feedforward → add & norm
        x = self.layer_norm2(x + self.ffn(x))
        
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # 2 MHAs, 3 LayerNorms, 1 FeedForward, 1 Dropout
        self.mha1 = MultiHeadAttention(d_model,num_heads)
        self.mha2 = MultiHeadAttention(d_model,num_heads)

        self.ffn = FeedForward(d_model,d_ff)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_out, src_mask=None, tgt_mask=None):
        # masked self attention → add & norm
        x = self.ln1( x + self.mha1(x,x,x,tgt_mask))
        # cross attention → add & norm
        x = self.ln2( x + self.mha2(x,encoder_out,encoder_out,src_mask))
        # feedforward → add & norm
        x = self.ln3( x + self.ffn(x))

        return x
    
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, 
                 num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        
        # embeddings — one for source, one for target
        # hint: nn.Embedding(vocab_size, d_model)
        self.src_embedding = nn.Embedding(src_vocab,d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab,d_model)
        
        # positional encoding — one shared is fine
        self.pe = PositionalEncoding(d_model)
        
        # stack of N encoder blocks
        # hint: nn.ModuleList([EncoderBlock(...) for _ in range(num_layers)])
        self.encoder_layers = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        
        # stack of N decoder blocks
        self.decoder_layers = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        
        # final linear projection → vocab size
        # hint: maps d_model → tgt_vocab
        self.fc_out = nn.Linear(d_model,tgt_vocab)
        
        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        
        # step 1 — embed + positional encode src
        src = self.src_embedding(src) 
        src = self.pe(src)
        src = self.dropout(src)
        
        # step 2 — pass through encoder stack
        for layer in self.encoder_layers:
            src = layer(src,src_mask)
        
        # step 3 — embed + positional encode tgt
        tgt = self.tgt_embedding(tgt)
        tgt = self.pe(tgt)
        tgt = self.dropout(tgt)
        
        # step 4 — pass through decoder stack
        # remember decoder needs both tgt and encoder output (src)
        for layer in self.decoder_layers:
            tgt = layer(tgt,src,src_mask,tgt_mask)
        
        # step 5 — final linear projection
        output = self.fc_out(tgt)
        
        return output
    
    