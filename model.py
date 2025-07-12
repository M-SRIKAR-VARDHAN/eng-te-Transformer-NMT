import torch 
import math 
import torch.nn as nn 

class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model= d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self , d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term  = torch.exp(torch.arange(0,d_model,2).float() *(-math.log(10000)/d_model))

        pe[:,0::2] = torch.sin(position*div_term )
        pe[:,1::2] = torch.cos(position*div_term )

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self,x):
        x = x + (self.pe[:,:x.shape[1],:])
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self, epsilon = 1e-6 ):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta =  nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim = -1 , keepdim = True)
        var = x.var(dim = -1 , keepdim = True)
        return (self.alpha * ((x-mean) / torch.sqrt(var + self.epsilon)))+self.beta

class FeedForwardBlock(nn.Module):
    def __init__(self , d_model , dff, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(d_model, dff)
        self.linear_2 = nn.Linear(dff , d_model)
    def forward(self , x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self , d_model , h , dropout ):
        super().__init__()
        self.d_model = d_model
        self.h = h 
        self.dropout = nn.Dropout (dropout)
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model , d_model)
        self.w_k = nn.Linear(d_model , d_model)
        self.w_v = nn.Linear(d_model , d_model)
        self.w_o = nn.Linear(d_model , d_model)
        # (batch_size , seq_len , d_model)
    @staticmethod
    def attention(query , key , value , mask , dropout ):
        # (batch_size , h , seq_len ,d_k)
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2,-1))/ math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax( dim = -1 )
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value),attention_scores

    def forward (self , q , k , v , mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)
        x , self.attention_scores =  MultiHeadAttention.attention(query , key ,value ,mask , self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h *self.d_k)

        return self.w_o(x)

class Residual(nn.Module):
    def __init__(self , dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x , sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block , feed_forward_layer , dropout):
        super().__init__()
        self.attention_block = self_attention_block 
        self.feed_forward_layer = feed_forward_layer
        self.residual = nn.ModuleList(Residual(dropout) for _ in  range(2))
        
    def forward (self , x ,src_mask ):
        x = self.residual[0](x,lambda x :self.attention_block(x,x,x,src_mask))
        x = self.residual[1](x , self.feed_forward_layer)

        return x
class Encoder(nn.Module):
    def __init__(self , layers ):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward (self ,x ,mask):
        for layer in self.layers:
            x = layer(x,mask)

        return self.norm(x)

class DecoderBlock (nn.Module):
    def __init__(self , self_attention_block , cross_attention_block ,feed_forward_layer , dropout ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_layer = feed_forward_layer 
        self.cross_attention_block = cross_attention_block 
        self.residual = nn.ModuleList(Residual(dropout) for _ in range(3))

    def forward (self , x  ,encoder_output , src_mask , trgt_mask  ):
        x = self.residual[0](x, lambda x:self.self_attention_block(x , x, x , trgt_mask) )
        x = self.residual[1](x , lambda x : self.cross_attention_block(x,encoder_output ,encoder_output ,src_mask))
        x = self.residual[2](x , self.feed_forward_layer)

        return x

class Decoder(nn.Module):
    def __init__(self , layers):
        super().__init__()
        self.layers = layers 
        self.norm = LayerNormalization(1e-6)

    def forward (self , x, encoder_output , src_mask , trgt_mask ):
        for layer in self.layers:
            x = layer(x, encoder_output , src_mask , trgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self , d_model , vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model , vocab_size)

    def forward( self , x):
        return torch.log_softmax(self.proj(x) , dim =-1)

class Transformer ( nn.Module):
    def __init__ (self , encoder , decoder , src_embed , trgt_embed , src_pos ,trgt_pos , projection_layer):
        super().__init__()
        self.encoder = encoder 
        self.decoder = decoder 
        self.src_embed = src_embed 
        self.trgt_embed = trgt_embed 
        self.src_pos = src_pos 
        self.trgt_pos = trgt_pos 
        self.projection_layer = projection_layer 

    def encode (self , src , src_mask ):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)

    def decode (self,encoder_output ,src_mask , trgt ,trgt_mask):
        trgt = self.trgt_embed(trgt)
        trgt = self.trgt_pos(trgt)
        return self.decoder(trgt, encoder_output,src_mask , trgt_mask)

    def project (self , x):
        return self.projection_layer (x)

def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model = 512, N = 6, h = 8, d_ff = 2048, dropout = 0.1) :

    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block,
            dropout
        )
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Assemble the full Transformer model
    transformer = Transformer( encoder, decoder, src_embed, tgt_embed,src_pos, tgt_pos, projection_layer )

    # Initialize parameters with Xavier uniform
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
    