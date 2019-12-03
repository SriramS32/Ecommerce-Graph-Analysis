# Bipartite random walks for link prediction

import os
import numpy as np
import pandas as pd
import networkx as nx
from utils import *
from math import floor
from tqdm import tqdm
from itertools import combinations

# ratio of frames to use to initialize the adjacency matrix
FRAME_RATIO = 0.1
TRAIN_SPLIT = 0.5

def init_adjacency_matrix(dat, TG, bins, frame_ratio=0.8):
    '''
    Generate an initial probability distribution of customer purchases
    as a C x P matrix
    
    Parameters:
       dat: pd.DataFrame containing retail data
       TG: temporal graph containing graph frames
       bins: pd.DatetimeIndex, keys to TG.frames
       frame_ratio: fraction of the bins to use
    '''

    adj = np.zeros((C,P))

    # number of frames to use
    num_frames = int(floor(len(bins) * 0.8))
    for date in bins[:num_frames]:
        date_key = str(date)
        cur_graph = TG.get_frame(date_key)
        for edge in cur_graph.edges:
            cID, sCode = extract_codes(edge)

            customer_idx, product_idx = node2idx[cID], node2idx[sCode]
            adj[customer_idx, product_idx] += 1

    return adj

if __name__ == '__main__':
    dat = read_retail_csv()

    # date bins
    bins = pd.date_range(start=dat.InvoiceDate.min(), end=dat.InvoiceDate.max(), freq='W')
    # keep only stock code and description, sort by stock code
    duplicated = dat[['StockCode', 'Description']].drop_duplicates().sort_values('StockCode')

    # generate separate indexing system for customerIDs and stockCodes
    customerIDs, stockCodes, node2idx = gen_indices(dat)
    C = len(customerIDs)
    P = len(stockCodes)

    TG = init_temporal_graph(dat, bins)
    
    print('Generating adjacency matrix...')
    adj = init_adjacency_matrix(node2idx, TG, bins, frame_ratio=FRAME_RATIO)
    normalize_rows(adj)
    print('Done!')

    num_frames = int(floor(len(bins) * FRAME_RATIO))
    train_limit = int(floor(len(bins) * TRAIN_SPLIT))
    print('Taking walk through the weeks...')
    # takes about 3 minutes for 50% of the frames
    for date in tqdm(bins[num_frames:train_limit], dynamic_ncols=True):
        date_key = str(date)
        cur_graph = TG.get_frame(date_key)
        T = gen_transition(cur_graph, stockCodes, node2idx)
        adj = normalize_rows(np.matmul(adj, T))
    print('Done!')

    # predicted_purchases = [np.argmax(adj[i, :]) for i in range(len(customerIDs))]
    # turns out, everyone's gonna buy birthday cards and christmas lights

    # now, calclulate precision!
    # scheme: (true positives / (true positive + true negative))
    print('Calculating precision...')
    prec_dict = calc_prec(adj, customerIDs, node2idx, TG, bins, TRAIN_SPLIT)
    print('Done!')
    avg_prec = np.mean([x for x in prec_dict.values()])
    print('Average customer precision: %f' % avg_prec)
    # average precision: 0.018747071735931934
    # median precision: 0.019954884755429024
    # best precision: 0.12178155639214931
    # worst precision: 0.0
