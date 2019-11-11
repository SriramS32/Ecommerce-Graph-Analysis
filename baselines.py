import os
import numpy as np
import pandas as pd
import networkx as nx
from utils import *
from math import floor
from tqdm import tqdm
from itertools import combinations

TRAIN_SPLIT = 0.8

def gen_partial_graph(TG, bins, r=0.8):
    '''
    Generate a simple graph between customers and products
      - NOTE: does not take advantage of multigraph structure
    '''
    num_frames = int(floor(len(bins) * r))
    
    G = nx.Graph()
    
    for date in bins[num_frames:-1]:
        date_key = str(date)
        cur_graph = TG.get_frame(date_key)
        G = nx.compose(G, cur_graph)

    return G

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

    partial_graph = gen_partial_graph(TG, bins, TRAIN_SPLIT)
    pairs = []
    print('Generating customer/product pairs...')
    # takes about 11 seconds on my computer
    for cID in tqdm(customerIDs, dynamic_ncols=True):
        if not partial_graph.has_node(cID):
            continue
        for sc in stockCodes:
            if not partial_graph.has_node(sc):
                continue
            if not partial_graph.has_edge(cID, sc):
                pairs.append((cID, sc))
    print('Done!')
    adj = np.zeros((C,P))
    probs = nx.preferential_attachment(partial_graph, pairs)
    print('Processing predictions...')
    # RAI takes about 12 minutes for me, 9,933,917 items w/ 0.8 ratio
    # Preferential attachment takes < 1 minute
    for u, v, p in tqdm(probs, dynamic_ncols=True):
        i = node2idx[u]
        j = node2idx[v]
        adj[i,j] = p
    print('Done!')

    adj = normalize_rows(adj)
    # preds = [np.argmax(adj[i, :]) for i in range(len(customerIDs))]

    prec_dict = calc_prec(adj, customerIDs, node2idx, TG, bins, TRAIN_SPLIT)
    avg_prec = np.mean([x for x in prec_dict.values()])
    print('Average customer precision = %.4f' % avg_prec)
        
