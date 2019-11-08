# Bipartite random walks for link prediction
#

import os
import numpy as np
import pandas as pd
import networkx as nx
from ..utils import *
from math import floor
from tqdm import tqdm
from itertools import combinations

# ratio of frames to use to initialize the adjacency matrix
FRAME_RATIO = 0.2

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
    
def gen_transition(G, stockCodes, node2idx):
    '''
    Generate a P x P transition matrix from the current graph,
    according to the following scheme:
       T_ii = 1
       T_ij = 
          - if i,j purchased together by at least one customer:
               # co-purchases / # purchases of item j
          - otherwise: 1/P
    
    Parameters:
       - G: a nx.Graph of the transactions in the current time frame
       - stockCodes: a list of customerIDs
       - node2idx
    '''

    P = len(stockCodes)
    initial_prob = 1.0 / P
    T = np.ones((P,P)) * initial_prob

    # map customerID -> purchased items
    customer_map = dict()
    # map purchased item -> customerIDs of purchased items
    product_map = dict()

    for edge in G.edges:
        cID, sCode = extract_codes(edge)
        if cID not in customer_map:
            customer_map[cID] = set()
        if sCode not in product_map:
            product_map[sCode] = set()
        customer_map[cID].add(sCode)
        product_map[sCode].add(cID)

    mapped_pairs = set()

    for purchase_record in customer_map.values():
        # every sorted combination of items
        for itemA, itemB in combinations(purchase_record, r=2):
            if (itemA, itemB) in mapped_pairs:
                continue
            mapped_pairs.add((itemA, itemB))
            setA = product_map[itemA]
            setB = product_map[itemB]
            intersection = len(setA.intersection(setB))

            probA = float(intersection) / len(setA)
            probB = float(intersection) / len(setB)

            idxA = node2idx[itemA]
            idxB = node2idx[itemB]

            T[idxA, idxB] = probB
            T[idxB, idxA] = probA

    for sCode in stockCodes:
        # all diagonals are 1.0
        idx = node2idx[sCode]
        T[idx, idx] = 1.0

    return T

def calc_prec(predictions, customerIDs, node2idx, TG, bins):
    '''
    Calculate a very coarse precision metric for 
    '''
    customer_map = dict()
    num_frames = int(floor(len(bins) * FRAME_RATIO))
    for date in tqdm(bins[num_frames:-1]):
        date_key = str(date)
        cur_graph = TG.get_frame(date_key)
        for cID in customerIDs:
            if cur_graph.has_node(cID):
                if cID not in customer_map:
                    customer_map[cID] = set()
                for sc in cur_graph.neighbors(cID):
                    customer_map[cID].add(node2idx[sc])

    total_purchased = len(customer_map)
    correct = 0
    for i, pred in enumerate(predictions):
        cID = customerIDs[i]
        if cID in customer_map and pred in customer_map[cID]:
            correct += 1

    return float(correct) / total_purchased
    

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
    print('Taking walk through the weeks...')
    # takes about 3 minutes for 50% of the frames
    for date in tqdm(bins[num_frames:-1], dynamic_ncols=True):
        date_key = str(date)
        cur_graph = TG.get_frame(date_key)
        T = gen_transition(cur_graph, stockCodes, node2idx)
        adj = normalize_rows(np.matmul(adj, T))
    print('Done!')

    predicted_purchases = [np.argmax(adj[i, :]) for i in range(len(customerIDs))]
    # turns out, everyone's gonna buy birthday cards and christmas lights

    # now, calclulate precision!
    # scheme: (true positives / (true positive + true negative))
    print('Calculating precision...')
    final_prec = calc_prec(predicted_purchases, customerIDs, node2idx, TG, bins)
    print('Done!')
    print('Final precision: %.4f' % final_prec)
                
                
