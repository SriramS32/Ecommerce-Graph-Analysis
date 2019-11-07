import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import sklearn
import os

from temporal import TemporalGraph
from utils import create_frame

TG_FRAME_PATH = './data/tg_frames'

if __name__ == '__main__':
    dat = pd.read_csv('data/online_retail_II.csv', parse_dates=['InvoiceDate'])
    dat = dat.dropna(subset=['StockCode', 'CustomerID'], how='any', axis=0)
    dat = dat[~dat.StockCode.isin(['M', 'POST', 'D', 'DOT', 'CRUK', 'C2', 'BANK CHARGES', 'ADJUST', 'ADJUST2', 'TEST001'])]

    # date bins
    bins = pd.date_range(start=dat.InvoiceDate.min(), end=dat.InvoiceDate.max(), freq='W')
    # keep only stock code and description, sort by stock code
    duplicated = dat[['StockCode', 'Description']].drop_duplicates().sort_values('StockCode')

    G = nx.MultiGraph()
    # some stock codes have letters in them
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

    TG = TemporalGraph((customerIDs, stockCodes), None)

    if os.path.isdir(TG_FRAME_PATH) and (len(os.listdir(TG_FRAME_PATH)) == (len(bins) - 1)):
        TG.read_pickles(TG_FRAME_PATH)
    else:

        # this loop takes about 3 minutes
        for i in tqdm(range(len(bins) - 1), dynamic_ncols=True):
            start = bins[i]
            end = bins[i+1]
            bin_dat = dat[(dat.InvoiceDate >= start) & (dat.InvoiceDate < end)]
            frame = create_frame(bin_dat)
            TG.add_frame(start, frame)
            
    print('Done loading temporal graph frame')
