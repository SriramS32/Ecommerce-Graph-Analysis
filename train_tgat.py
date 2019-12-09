import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pickle
import time
from math import ceil, floor
from tqdm import tqdm
from utils import *
from tgat import EcommerceTransformer
from adj_dataset import AdjacencyMatrixDataset

# profiling tools
from pytorch_memlab import profile, MemReporter
ADJ_SAVEPATH = './data/sparse_tensors'
MODEL_SAVEPATH = './data/transformer_model'
EMBEDDINGS = {
    'spectral': get_spectral_embedding,
    'learned': None,
    'random': get_random_embedding
}

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def sparse_narrow(data, nbatch, bsz):
    idxs = data._indices()
    vals = data._values()

    valid_idxs = []
    for i, timestep in enumerate(idxs[0]):
        if timestep.item() < nbatch * bsz:
            valid_idxs.append(i)

    new_idxs = torch.LongTensor([[idxs[j, i].item() for j in range(3)] for i in valid_idxs])
    new_vals = torch.FloatTensor([vals[i].item() for i in valid_idxs])
    return torch.sparse.FloatTensor(new_idxs.t(), new_vals,
                                    torch.Size([nbatch*bsz, data.size(1), data.size(2)]))

def make_seqs(data, bsz):
    seq_len = data.size(0)
    per_dim = seq_len // bsz
    idxs = data._indices()
    seq_idxs = [x.item() for x in data._indices()[0,:]]
    vals = data._values()

    new_idxs = torch.LongTensor([[i // per_dim, i % per_dim] +
                                 [idxs[j+1,i].item() for j in range(2)]
                                 for i in seq_idxs])
    return torch.sparse.FloatTensor(new_idxs.t(), vals,
                                    torch.Size([bsz, per_dim, data.size(1), data.size(2)]))

def batchify(data, C, P, bsz, device='cpu'):
    nbatch = data.size(0) // bsz
    # data = data.narrow(0, 0, nbatch * bsz)
    data = sparse_narrow(data, nbatch, bsz)
    # S x B x C x P
    data = make_seqs(data, bsz)
    data = data.transpose(0,1)
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

def split_data(args, bins, C, P, node2idx, TG):
    '''
    Generate the embedding sequence from the temporal graph
    and the product embedding
    '''

    print('Loading graph adjacency tensors...')

    train_idxs = []
    train_counts = []
    test_idxs = []
    test_counts = []

    cutoff = floor(0.8 * len(bins))

    for i, date in enumerate(bins[:-1]):
        date_key = str(date)
        cur_graph = TG.get_frame(date_key)
        edges = dict()
        for edge in cur_graph.edges:
            cID, sCode = extract_codes(edge)
            coords = node2idx[cID], node2idx[sCode]
            if coords in edges:
                edges[coords] += 1
            else:
                edges[coords] = 1

        if len(edges) == 0:
            continue
        else:
            if i < cutoff:
                train_idxs.extend([[i] + list(coords) for coords, _ in edges.items()])
                train_counts.extend([count for _, count in edges.items()])
            else:
                test_idxs.extend([[i-cutoff] + list(coords) for coords, _ in edges.items()])
                test_counts.extend([count for _, count in edges.items()])

    train_idxs = torch.LongTensor(train_idxs)
    train_counts = torch.FloatTensor(train_counts)
    train_split = torch.sparse.FloatTensor(train_idxs.t(), train_counts,
                                           torch.Size([cutoff, C, P]))

    test_idxs = torch.LongTensor(test_idxs)
    test_counts = torch.FloatTensor(test_counts)
    test_split = torch.sparse.FloatTensor(test_idxs.t(), test_counts,
                                          torch.Size([len(bins) - cutoff, C, P]))

    # S x C x P
    train_split = batchify(train_split,
                           C, P,
                           args.batch_size
    )

    test_split = batchify(test_split,
                          C, P,
                          args.batch_size
    )

    if not os.path.exists(ADJ_SAVEPATH):
        os.makedirs(ADJ_SAVEPATH)
    print('Saving tensors...')
    torch.save(train_split._indices(), ADJ_SAVEPATH + '/batch%d_train_indices.pt' % args.batch_size)
    torch.save(train_split._values(), ADJ_SAVEPATH + '/batch%d_train_values.pt' % args.batch_size)
    torch.save(test_split._indices(), ADJ_SAVEPATH + '/batch%d_test_indices.pt' % args.batch_size)
    torch.save(test_split._values(), ADJ_SAVEPATH + '/batch%d_test_values.pt' % args.batch_size)

    print('Done!')

    return train_split, test_split

def path_ok(bsz):
    check1 = os.path.exists(ADJ_SAVEPATH)
    if not check1:
        return False
    check2 = os.path.exists(ADJ_SAVEPATH + '/batch%d_train_indices.pt' % bsz)
    check3 = os.path.exists(ADJ_SAVEPATH + '/batch%d_test_indices.pt' % bsz)
    check4 = os.path.exists(ADJ_SAVEPATH + '/batch%d_train_values.pt' % bsz)
    check5 = os.path.exists(ADJ_SAVEPATH + '/batch%d_test_values.pt' % bsz)
    return check2 and check3 and check4 and check5

def get_split(args, bins, C, P, node2idx, TG, split='train'):
    assert split in ['train', 'test']
    if path_ok(args.batch_size):
        indices = torch.load(os.path.join(ADJ_SAVEPATH, ('batch%d_%s_indices.pt' %
                                                         (args.batch_size, split))))
        values = torch.load(os.path.join(ADJ_SAVEPATH, ('batch%d_%s_values.pt' %
                                                        (args.batch_size, split))))
        cutoff = floor(len(bins) * 0.8)
        seq_len = cutoff // args.batch_size if split == 'train' \
            else (len(bins) - cutoff) // args.batch_size
        return torch.sparse.FloatTensor(indices, values,
                                        torch.Size([seq_len, args.batch_size, C, P]))
    else:
        trainset, testset = split_data(args, bins, C, P, node2idx, TG)
        if split == 'train':
            del testset
            return trainset
        else:
            del trainset
            return testset

def narrow_slice(src, lb, ub):
    idxs = src._indices()
    vals = src._values()

    valid_idxs = []
    for i, timestep in enumerate(idxs[0]):
        ts = timestep.item()
        if lb <= ts and ts < ub:
            valid_idxs.append(i)

    new_idxs = torch.LongTensor([
        [idxs[0, i] - lb] + [idxs[j+1, i].item() for j in range(3)] for i in valid_idxs
    ])
    new_vals = torch.FloatTensor([vals[i].item() for i in valid_idxs])
    return torch.sparse.FloatTensor(new_idxs.t(), new_vals,
                                    torch.Size([ub-lb, src.size(1), src.size(2), src.size(3)]))

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

    for i in range(N-seq_len):
        src = narrow_slice(src_tensor, i, i+seq_len)
        target = narrow_slice(src_tensor, i+1, i+seq_len+1)
        src.coalesce()
        target.coalesce()
        srcs.append(src)
        targets.append(target)

    return AdjacencyMatrixDataset(srcs, targets)

def sparse_zero_index(t):
    idxs = t._indices()
    vals = t._values()

    new_idxs = torch.LongTensor([
        [coord[i+1] for i in range(4)]
        for coord in idxs.t()])
    return torch.sparse.FloatTensor(new_idxs.t(), vals,
                                    t.shape[1:])
    
def train(model,
          trainloader,
          optimizer,
          epochs,
          embedding_matrix,
          criterion=nn.MSELoss(),
          seq_length=4,
          log_interval=20,
          device='cpu'):
    model.train()
    total_loss = 0
    start_time = time.time()
    for epoch in range(epochs):
        for i, (src, target) in enumerate(trainloader):
            if 'cuda' in device:
                src = sparse_zero_index(src).cuda()
                target_emb = torch.matmul(sparse_zero_index(target).cuda().to_dense(),
                                             embedding_matrix)
            optimizer.zero_grad()
            output = model(src)
            # need to create target output from the target sequence
            C = output.shape[2] // embedding_matrix.shape[1]
            output = output.view(seq_length, output.shape[1],
                                 C, embedding_matrix.shape[1])
        
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
                          i, len(trainloader),
                          elapsed * 1000 / log_interval,
                          avg_loss
                      ))
                total_loss = 0
                start_time = time.time()

def evaluate(model,
             testloader,
             embedding_matrix,
             seq_length,
             C, P,
             device='cpu'):

    # make predictions on each of the batches, generate a prediction matrix, and
    # calculate average hit rate

    print('Evaluating...')
    totals = torch.zeros((C,P))
    probs = torch.zeros((C,P))
    model.eval()
    with torch.no_grad():
        for (src, target) in tqdm(testloader):
            target = target.to_dense()
            totals += torch.sum(target[0], dim=[0,1])
            if 'cuda' in device:
                model.cuda()
                src = sparse_zero_index(src).cuda()
            # S x B x C x D
            output = model(src)
            if 'cuda' in device:
                # move it back to save memory
                model.to('cpu')
                src = src.to('cpu')
            C = output.shape[2] // embedding_matrix.shape[1]
            output = output.view(seq_length, output.shape[1],
                                 C, embedding_matrix.shape[1])
            preds = torch.matmul(output, embedding_matrix.t())
            # sigmoid, without all the baggage of the torch function
            preds = 1 / (1 + torch.exp(-preds.to('cpu')))
            probs += torch.sum(preds * target[0], dim=[0,1])

    # calculate prec the same way as for the rw baseline
    correct_total = torch.matmul(probs, torch.ones((P,)))
    customer_total = torch.matmul(totals, torch.ones((P,)))
    customer_prec = correct_total / customer_total
    # if x is Nan, `x != x` == True, so we check for Nans here
    customer_prec = customer_prec.masked_fill(customer_prec != customer_prec, 0)

    avg_prec = (torch.sum(customer_prec) / len(customer_prec)).item()

    print('Done! Avg precision = %.4f' % avg_prec)

    return avg_prec
        

def gen_embedding_matrix(bins, TG, stockCodes, node2idx,
                         embedding_type='spectral',
                         embedding_dim=8):
    n = len(bins)
    emb = None
    emb_f = EMBEDDINGS[embedding_type]
    filepath = './data/product_embeddings/%s_dim%d.npy' % (embedding_type, embedding_dim)

    if emb_f is None:
        return None

    if os.path.exists(filepath):
        emb = np.load(filepath)
    else:
        # no entry for the last point
        for date in bins[:-1]:
            date_key = str(date)
            cur_graph = TG.get_frame(date_key)
            if emb is None:
                emb = (emb_f(cur_graph, stockCodes, node2idx, embedding_dim) / n )
            else:
                emb = emb + ( emb_f(cur_graph, stockCodes, node2idx, embedding_dim) / n )
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, emb)
    return emb

def main(args):
    set_random_seed(args.seed)
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
    model_args = [C, P, args.embedding_dim, args.num_heads, args.hidden,
                  args.num_layers]
    model = EcommerceTransformer(*model_args,
                                 embedding=embedding_matrix)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    if 'cuda' in args.device:
        model.cuda()
    
    # create the dataset
    trainset = get_split(args, bins, C, P, node2idx, TG, split='train')
    trainset = build_dataset(trainset, args.batch_size, args.seq)
    # splits are already split into batches
    trainloader = torch.utils.data.DataLoader(trainset,
                                              num_workers=args.num_workers,
                                              pin_memory=False)
    
    # convert to tensor to give to model
    if embedding_matrix is not None:
        embedding_matrix = torch.FloatTensor(embedding_matrix)
        if 'cuda' in args.device:
            embedding_matrix = embedding_matrix.cuda()
        
        
    train(model,
          trainloader,
          optimizer,
          args.epochs,
          embedding_matrix,
          criterion=nn.MSELoss(),
          seq_length=args.seq,
          log_interval=args.log_interval,
          device=args.device)

    print('Finished training! Saving to file...')
    # b = batch size
    # s = seq length
    # h = number of heads
    # l = number of layers
    # hid = hidden size
    model_spec_str = '%s_b%d_s%d_h%d_l%d_hid%d' % \
        (args.embedding_type, args.batch_size, args.seq,
         args.num_heads, args.num_layers, args.hidden)
    savepath = os.path.join(MODEL_SAVEPATH, model_spec_str + '.pt')
    arg_path = os.path.join(MODEL_SAVEPATH, model_spec_str + '_ARGS.p')
    if not os.path.exists(MODEL_SAVEPATH):
        os.makedirs(MODEL_SAVEPATH)
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, savepath)
    with open(arg_path, 'wb') as f:
        pickle.dump(model_args, f)
    print('Saved to `%s`, done!' % savepath)

    del trainloader
    del trainset

    testset = get_split(args, bins, C, P, node2idx, TG, split='test')
    testset = build_dataset(testset, args.batch_size, args.seq)
    testloader = torch.utils.data.DataLoader(testset,
                                            num_workers=args.num_workers,
                                            pin_memory=False)

    evaluate(model, testloader, embedding_matrix, args.seq, C, P, device=args.device)

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
                        choices=['spectral', 'random', 'learned'], help='Type of initial graph embedding to use')
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
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of dataloader workers')

    args = parser.parse_args()
    args.device = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'

    main(args)
