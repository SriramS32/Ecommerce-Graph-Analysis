import pandas as pd
import numpy as np
import networkx as nx
import os
from tqdm import tqdm
from itertools import combinations
from math import floor
from temporal import TemporalGraph
from sklearn.manifold import SpectralEmbedding, TSNE

CSV_PATH = './data/online_retail_II.csv'
TG_FRAME_PATH = './data/tg_frames'
# codes to be dropped from the dataframe
MANUAL_CODES = ['M', 'POST', 'D', 'DOT', 'CRUK', 'C2', 'BANK CHARGES',
                'ADJUST', 'ADJUST2', 'TEST001']

def read_retail_csv():
    '''
    Load the Online Retail II dataset from the CSV, drop some manual stock codes
    and delete the nan values
    '''
    dat = pd.read_csv(CSV_PATH, parse_dates=['InvoiceDate'])
    dat = dat.dropna(subset=['StockCode', 'CustomerID'], how='any', axis=0)
    dat = dat[~dat.StockCode.isin(MANUAL_CODES)]

    return dat

def create_frame(dat):
    '''
    Create a graph that captures the purchase information within a single time frame
    
    NOTE: ignores multigraph structure here, e.g. if customer A makes 2 separate purchases
    of the same item, only 1 unweighted edge will be included
    '''
    
    G = nx.Graph()
    
    #Index(['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate',
    #   'Price', 'CustomerID', 'Country'],
    #  dtype='object')

    custData = dat[['CustomerID', 'Country']]\
        .drop_duplicates(subset=['CustomerID'], keep='last')
    stockData = dat[['StockCode', 'Description']].drop_duplicates(subset=['StockCode'], keep='last')
    
    assert len(custData) == len(custData.CustomerID.unique())
    assert len(stockData) == len(stockData.StockCode.unique())
    
    for i, stock in stockData.iterrows():
        G.add_node(stock.StockCode,
                   description=stock.Description)
    
    for i, customer in custData.iterrows():
        G.add_node(int(customer.CustomerID),
                   country=customer.Country)
        
    for i, row in dat.iterrows():
        G.add_edge(row.StockCode,
                   int(row.CustomerID),
                   key=row.Invoice,
                   quantity=row.Quantity,
                   date=row.InvoiceDate,
                   price=row.Price) # sometimes price changes, sale?
        
    return G

def init_temporal_graph(dat, bins):
    '''
    Generate the temporal graph object from the pandas file and the date bins
    '''
    print('Loading temporal graph frames...')
    # currently the two parameters to TemporalGraph don't do anything
    TG = TemporalGraph(None, None)

    if os.path.isdir(TG_FRAME_PATH) and (len(os.listdir(TG_FRAME_PATH)) == (len(bins) - 1)):
        print('Found gpickles, loading from gpickles...')
        TG.read_gpickles(TG_FRAME_PATH)
    else:
        # create the parent folder, along with any intermediate folders
        os.makedirs(TG_FRAME_PATH, exist_ok=True)
        # this loop takes about 3 minutes
        for i in tqdm(range(len(bins) - 1), dynamic_ncols=True):
            start = bins[i]
            end = bins[i+1]
            bin_dat = dat[(dat.InvoiceDate >= start) & (dat.InvoiceDate < end)]
            frame = create_frame(bin_dat)
            TG.add_frame(start, frame)

        TG.write_gpickles(TG_FRAME_PATH)

    print('Done!')

    return TG

def gen_indices(dat):
    '''
    Generate a separate indexing system for customerIDs and stockCodes
    '''
    G = nx.MultiGraph()
    # some stock codes have letters in them, so we keep it as a string
    edges = list(zip(dat.StockCode, dat.CustomerID.astype(int)))
    for edge in edges:
        G.add_edge(edge[0], edge[1])

    customerIDs = []
    stockCodes = []
    node2idx = dict()

    for node in G.nodes:
        if type(node) == str:
            # we have a stock code
            idx = len(stockCodes)
            node2idx[node] = idx
            stockCodes.append(node)
        else:
            # we have a customer ID
            assert(type(node) == int)
            idx = len(customerIDs)
            node2idx[node] = idx
            customerIDs.append(node)

    return customerIDs, stockCodes, node2idx

def extract_codes(edge):
    '''
    Given an edge (u,v), determine which one is a customerID and
    which one is a stock code and return them in order
    '''

    u = edge[0]
    v = edge[1]

    if type(u) == str:
        assert type(v) == int , 'u has type str but v has type %s' % str(type(v))
        customerID = v
        stockCode = u
    else:
        assert type(u) == int and type(v) == str, \
            'u, v have types %s, %s' % (str(type(u)), str(type(v)))
        customerID = u
        stockCode = v
    return customerID, stockCode

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
        # every sorted combination of items purchased by each customer
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

            T[idxA, idxB] = probA
            T[idxB, idxA] = probB

    for sCode in stockCodes:
        # all diagonals are 1.0
        idx = node2idx[sCode]
        T[idx, idx] = 1.0

    return T

def normalize_rows(A):
    '''
    normalize the rows of A
    '''
    norm = np.linalg.norm(A, ord=2, axis=1, keepdims=True)
    # don't divide by 0!
    norm[norm == 0] = 1.0
    return A / norm

def calc_prec(adj, customerIDs, node2idx, TG, bins, ratio):
    '''
    Calculate a very coarse precision metric for precision.
        We only consider 'valid' customers that bought at least 1 item:
        A prediction is correct if the 1 predicted item
        is in the total set of items purchased by the customers.
        The metric is then # correct / # valid customers 
    '''
    customer_map = dict()
    num_frames = int(floor(len(bins) * ratio))
    for date in tqdm(bins[num_frames:-1]):
        date_key = str(date)
        cur_graph = TG.get_frame(date_key)
        for cID in customerIDs:
            if cur_graph.has_node(cID):
                if cID not in customer_map:
                    customer_map[cID] = dict()
                for sc in cur_graph.neighbors(cID):
                    scIdx = node2idx[sc]
                    if scIdx not in customer_map[cID]:
                        customer_map[cID][scIdx] = 0
                    customer_map[cID][scIdx] += 1

    total_purchased = len(customer_map)
    correct = 0
    # map from cID to total_precision
    customer_prec = dict()
    for i in range(len(customerIDs)):
        customer_total = 0
        correct_total = 0
        cID = customerIDs[i]
        probs = adj[i, :]
        if cID not in customer_map:
            continue
        purchase_dict = customer_map[cID]
        for scIdx, pCount in purchase_dict.items():
            customer_total += pCount
            correct_total += (pCount * probs[scIdx])
        if customer_total > 0:
            customer_prec[cID] = (correct_total) / (customer_total)
    return customer_prec

def get_adj_matrix(G, node2idx, C, P):
    adj = np.zeros((C+P, C+P))

    for edge in G.edges:
        cID, sCode = extract_codes(edge)
        customer_idx, product_idx = node2idx[cID], node2idx[sCode]
        adj[customer_idx, C+product_idx] = 1
        adj[C+product_idx, customer_idx] = 1

    return adj

def tisne_embedding(G):
    '''
    Generates a TISNE embedding from the customer-product adjacency matrix
    '''

    raise Exception('Not implemented!')

def get_spectral_embedding(G, stockCodes, node2idx, dim=8, seed=1):
    '''
    Generates a spectral embedding from the networkx graph
       - embedding dimension of 8 is the fourth root of 4636 (length of stockCodes)
          - TODO: try different embedding dimensions
    '''
    
    # get the product adjacency matrix, won't be symmetric but will be converted to one
    adj = gen_transition(G, stockCodes, node2idx)
    
    model = SpectralEmbedding(n_components=dim, affinity='precomputed', random_state=seed,
                              n_neighbors=None, n_jobs=None)
    return model.fit_transform(adj)
