import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATEmbedding(nn.Module):
    '''
    Dense GAT as an embedding layer
    '''
    def __init__():
        super(GATEmbedding, self).__init__()

class EcommerceTransformer(nn.Module):
    '''
    Largely from the PyTorch transformer tutorial:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    '''

    def __init__(self, C, P, ninp, nhead, nhid, nlayers, dropout=0.5, embedding=None):
        '''
        ntoken = size of vocab, C*P in this case
        ninp = embedding dimension
        nhid = size of hidden layers
        nlayers = number of encoder layers
        '''
        super(EcommerceTransformer, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        ntoken = C * P
        self.C = C
        self.P = P
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(C*ninp, dropout)
        encoder_layers = TransformerEncoderLayer(C*ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.custom_embedding = False
        if embedding is not None:
            self.custom_embedding = True
            self.encoder = torch.FloatTensor(embedding)
        else:
            self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        # could replace decoder with the embedding layer as well
        # output: S x B x V, in our case V gives a customer embedding
        self.decoder = nn.Linear(self.C * ninp, self.C * ninp)

        self.init_weights()

    def cuda(self):
        super(EcommerceTransformer, self).cuda()
        if self.custom_embedding:
            self.encoder = self.encoder.cuda()
        if self.src_mask is not None:
            self.src_mask = self.src_mask.cuda()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        if not self.custom_embedding:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        if self.custom_embedding:
            seq = src.shape[0]
            batch = src.shape[1]
            src = torch.matmul(src.view(seq, batch, self.C, self.P), self.encoder)
            src = src.view(seq, batch, -1)
        else:
            src = self.encoder(src) * math.sqrt(self.ninp)
        # positional information
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
