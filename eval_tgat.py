import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from utils import *
from tgat import EcommerceTransformer
from train_tgat import set_random_seed, get_split, \
    gen_embedding_matrix, evaluate, build_dataset

def main(args):
    set_random_seed(args.seed)
    dat = read_retail_csv()
    bins = pd.date_range(start=dat.InvoiceDate.min(), end=dat.InvoiceDate.max(), freq='W')
    duplicated = dat[['StockCode', 'Description']].drop_duplicates().sort_values('StockCode')
    customerIDs, stockCodes, node2idx = gen_indices(dat)
    C = len(customerIDs)
    P = len(stockCodes)
    TG = init_temporal_graph(dat, bins)

    model_args_path = args.savepath.replace('.pt', '_ARGS.p')
    with open(model_args_path, 'rb') as f:
        model_args = pickle.load(f)

    # only one for now
    embedding_type = 'spectral'
    embedding_dim = model_args[2]
    
    # embedding matrix : P x D
    # right now just using spectral embedding of product transition matrix
    embedding_matrix = gen_embedding_matrix(bins, TG, stockCodes, node2idx,
                                                embedding_type,
                                                embedding_dim)

    print('Loading model...')
    model = EcommerceTransformer(*model_args, embedding=embedding_matrix)
    ckpt = torch.load(args.savepath)
    model.load_state_dict(ckpt['model_state_dict'])
    print('Done!')

    print('Loading test set...')
    testset = get_split(args, bins, C, P, node2idx, TG, split='test')
    testset = build_dataset(testset, args.batch_size, args.seq)
    testloader = torch.utils.data.DataLoader(testset,
                                            num_workers=args.num_workers,
                                            pin_memory=False)
    print('Done!')

    embedding_matrix = torch.FloatTensor(embedding_matrix)
    if 'cuda' in args.device:
        embedding_matrix = embedding_matrix.cuda()

    evaluate(model, testloader, embedding_matrix, args.seq,
             C, P, device=args.device)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of sequences '
                        'in one batch')
    parser.add_argument('--seq', type=int, default=4,
                        help='Sequence length')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of dataloader workers')
    parser.add_argument('--savepath', type=str, default=None,
                        required=True, help='path to saved model checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu','cuda'],help='Device to run on')
    args = parser.parse_args()
    args.device = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'

    main(args)

