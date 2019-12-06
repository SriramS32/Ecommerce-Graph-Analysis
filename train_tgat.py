import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time
from math import ceil, floor
from utils import *
from tgat import EcommerceTransformer
from adj_dataset import AdjacencyMatrixDataset

ADJ_SAVEPATH = './data/adj_tensors'
EMBEDDINGS = {
    'spectral': get_spectral_embedding
}

def batchify(data, C, P, bsz, device='cpu'):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    # S x B x C x P
    data = data.view(bsz, -1, C, P).transpose(0,1).contiguous()
    return data.to(device)

def get_batch(source, i, embedding_matrix, max_seq_len=4):
    batch = source.shape[1]
    P = embedding_matrix.shape[0]
    
    seq_len = min(max_seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target_adj = source[i+1:i+1+seq_len]
    norm_vec = 1 / torch.matmul(target_adj, torch.ones(P))
    norm_vec = norm_vec.masked_fill(norm_vec == float('inf'), 0.0)
    target = torch.matmul(target_adj, embedding_matrix) * norm_vec.view(seq_len, batch,-1,1)
    return data, target

def split_data(args, bins, C, P, node2idx, TG, embedding_matrix):
    '''
    Generate the embedding sequence from the temporal graph
    and the product embedding
    '''

    print('Loading graph adjacency tensors...')

    whole_sequence = []

    for date in bins[:-1]:
        date_key = str(date)
        cur_graph = TG.get_frame(date_key)
        cur_adj = np.zeros((C,P))
        for edge in cur_graph.edges:
            cID, sCode = extract_codes(edge)
            customer_idx, product_idx = node2idx[cID], node2idx[sCode]
            cur_adj[customer_idx, product_idx] += 1

        # emb = np.matmul(cur_adj, embedding_matrix)
        whole_sequence.append(cur_adj)

    # S x C x P
    whole_sequence = whole_sequence

    train_split = batchify(torch.FloatTensor(whole_sequence[: floor(0.8 * len(bins))]),
                           C, P,
                           args.batch_size,
                           device=args.device
    )

    val_split = batchify(torch.FloatTensor(whole_sequence[len(train_split):floor(0.9 * len(bins))]),
                         C,P,
                         args.batch_size,
                         device=args.device
    )
    test_split = batchify(torch.FloatTensor(whole_sequence[len(train_split) + len(val_split) : ]),
                          C, P,
                          args.batch_size,
                          device=args.device
    )

    if not os.path.exists(ADJ_SAVEPATH):
        os.makedirs(ADJ_SAVEPATH)
    print('Saving tensors...')
    torch.save(train_split, ADJ_SAVEPATH + '/train.pt')
    torch.save(val_split, ADJ_SAVEPATH + '/val.pt')
    torch.save(test_split, ADJ_SAVEPATH + '/test.pt')

    print('Done!')

    return train_split, val_split, test_split

def build_dataset(src_tensor, batch_size, seq_len):
    '''
    creates a torch.utils.data.Dataset of (src, target) pairs from a tensor
    - tensor has shape N x B x C x P
        - N = number of batches
        - B = batch size
        - C x P = size of adjacency matrix
    '''

    srcs = []
    targets = []

    N = src_tensor.shape[0]

    for i in range(N-seq_len-1):
        src = src_tensor[i:i+seq_len,:,:,:]
        target = src_tensor[(i+1):(i+seq_len+1),:,:,:]
        srcs.append(src)
        targets.append(target)

    return AdjacencyMatrixDataset(srcs, targets)
    

def train(model,
          trainloader,
          optimizer,
          embedding_matrix,
          criterion=nn.MSELoss(),
          seq_length=4,
          log_interval=20,
          device='cpu'):
    model.train()
    total_loss = 0
    start_time = time.time()
    for i, (src, target) in enumerate(trainloader):
        if 'cuda' in device:
            src = src.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = model(src[0])
        # need to create target output from the target sequence
        output = output.view(seq_length, output.shape[1],
                             output.shape[2], embedding_matrix.shape[1])
        target_emb = torch.matmul(target[0], embedding_matrix)
        loss = criterion(output, target_emb)
        loss.backward()
        # try not to use grad clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if i % log_interval == 0 and i > 0:
            avg_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches |'
                  'ms/batch {:5.2f} |'
                  'avg loss {:5.2f}'.format(
                      epoch,
                      i, len(srcs),
                      elapsed * 1000 / log_interval,
                      avg_loss
                  ))
            total_loss = 0
            start_time = time.time()

def gen_embedding_matrix(bins, TG, stockCodes, node2idx,
                         embedding_type='spectral',
                         embedding_dim=8,
                         seed=1):
    n = len(bins)
    emb = None
    emb_f = EMBEDDINGS[embedding_type]
    filepath = './data/product_embeddings/dim%d.npy' % embedding_dim

    if os.path.exists(filepath):
        emb = np.load(filepath)
    else:
        # no entry for the last point
        for date in bins[:-1]:
            date_key = str(date)
            cur_graph = TG.get_frame(date_key)
            if emb is None:
                emb = (emb_f(cur_graph, stockCodes, node2idx, embedding_dim, seed) / n )
            else:
                emb = emb + ( emb_f(cur_graph, stockCodes, node2idx, embedding_dim, seed) / n )
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, emb)
    return emb

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
    # embedding matrix : P x D
    # right now just using spectral embedding of product transition matrix
    embedding_matrix = gen_embedding_matrix(bins, TG, stockCodes, node2idx,
                                                args.embedding_type,
                                                args.embedding_dim)

    # create the model
    model = EcommerceTransformer(C,
                                 P,
                                 args.embedding_dim,
                                 args.num_heads,
                                 args.hidden,
                                 args.num_layers,
                                 embedding=embedding_matrix)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    if 'cuda' in args.device:
        model.cuda()
    
    # create the dataset
    if os.path.exists(ADJ_SAVEPATH) and len(os.listdir(ADJ_SAVEPATH)) == 3:
        print('Found saved tensors, loading from tensors...')
        trainset = torch.load(ADJ_SAVEPATH + '/train.pt')
        # valset = torch.load(ADJ_SAVEPATH + '/val.pt')
        # testset = torch.load(ADJ_SAVEPATH + '/test.pt')
    else:
        trainset, valset, testset = split_data(args, bins, C, P, node2idx,
                                           TG, embedding_matrix)
        # remove unnecessary splits to save memory for now
        # | we will retrieve them later from their savepath
        del valset
        del testset

    trainset = build_dataset(trainset, args.batch_size, args.seq)
    # splits are already split into batches
    trainloader = torch.utils.data.DataLoader(trainset,
                                              num_workers=args.num_workers,
                                              pin_memory=True)
    
    # convert to tensor to give to model
    embedding_matrix = torch.FloatTensor(embedding_matrix)
    if 'cuda' in args.device:
        embedding_matrix = embedding_matrix.cuda()
        
        
    train(model,
          trainloader,
          optimizer,
          embedding_matrix,
          criterion=nn.MSELoss(),
          seq_length=args.seq,
          log_interval=args.log_interval,
          device=args.device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay '
                        '(L2 regularization on parameters)')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of sequences '
                        'in one batch')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu','cuda'],help='Device to run on')
    parser.add_argument('--embedding-type', type=str, default='spectral',
                        choices=['spectral', ''], help='Type of initial graph embedding to use')
    parser.add_argument('--embedding-dim', type=int, default=8,
                        help='Size of the initial embedding dimension')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--hidden', type=int, default=200,
                        help='Size of the network hidden layers')
    parser.add_argument('--num-layers', type=int, default=8,
                        help='Number of transformer encoder layers')
    parser.add_argument('--seq', type=int, default=4,
                        help='Sequence length')
    parser.add_argument('--log-interval', type=int, default=20,
                        help='Period over which to log training results')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of dataloader workers')

    args = parser.parse_args()
    args.device = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'

    main(args)
