import os
import numpy as np
import pandas as pd
import networkx as nx
from utils import *
from math import floor
from tqdm import tqdm
from itertools import combinations

FRAME_RATIO = 0.1
TRAIN_SPLIT = 0.5

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

    uniform_probs = np.ones((C,P)) * (1.0 / P)
    print('Calculating precision...')
    prec_dict = calc_prec(uniform_probs, customerIDs, node2idx,
                          TG, bins, TRAIN_SPLIT)
    print('Done!')
    avg_prec = np.mean([x for x in prec_dict.values()])
    print('Average customer precision: %f' % avg_prec)
