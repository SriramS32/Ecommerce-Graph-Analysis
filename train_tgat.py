import torch
import torch.nn as nn
import pandas as pd
from utils import *

EMBEDDINGS = {
    'spectral': get_spectral_embedding
}

def batchify(data, bsz, device='cpu'):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def get_batch(source, i, bptt=35):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def train(model,
          srcs,
          targets,
          optimizer,
          scheduler,
          criterion=nn.CrossEntropyLoss(),
          bptt=35,
          log_interval=20):
    model.train()
    total_loss = 0
    start_time = time.time()
    for (src, target), i in enumerate(zip(srcs, targets)):
        optimizer.zero_grad()
        output = model(src)
        loss = criterion(output, target)
        loss.backward()
        # try not to use grad clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if i % log_interval == 0 and i > 0:
            avg_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches |'
                  'lr {:02.2f} | ms/batch {:5.2f} |'
                  'avg loss {:5.2f}'.format(
                      epoch,
                      i, len(srcs),
                      scheduler.get_lr()[0],
                      elapsed * 1000 / log_interval,
                      avg_loss
                  ))
            total_loss = 0
            start_time = time.time()

def gen_embedding_sequence(bins, TG, stockCodes, node2idx, embedding_type, embedding_dim, seed):
    data = []
    emb_f = EMBEDDINGS[embedding_type]

    for date in bins:
        date_key = str(date)
        cur_graph = TG.get_frame(date_key)
        data.append(emb_f(G, stockCodes, node2idx, embedding_dim, seed))

    return data
        

def main(args):
    dat = read_retail_csv()
    bins = pd.date_range(start=dat.InvoiceDate.min(), end=dat.InvoiceDate.max(), freq='W')
    # keep only stock code and description, sort by stock code
    duplicated = dat[['StockCode', 'Description']].drop_duplicates().sort_values('StockCode')
    # generate separate indexing system for customerIDs and stockCodes
    customerIDs, stockCodes, node2idx = gen_indices(dat)
    C = len(customerIDs)
    P = len(stockCodes)
    TG = init_temporal_graph(dat, bins)
    embedding_sequence = gen_embedding_sequence(bins, TG, stockCodes, node2idx,
                                                args.embedding_type,
                                                args.embedding_dim)

    # TODO:
    # generate total product embedding matrix B : P x D
