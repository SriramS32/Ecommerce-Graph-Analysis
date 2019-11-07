import pandas as pd
import numpy as np
import networkx as nx
import os
from tqdm import tqdm

from temporal import TemporalGraph


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

def normalize_rows(A):
    '''
    normalize the rows of A
    '''
    norm = np.linalg.norm(A, ord=2, axis=1, keepdims=True)
    # don't divide by 0!
    norm[norm == 0] = 1.0
    return A / norm
