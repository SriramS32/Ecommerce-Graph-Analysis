import pandas as pd
import numpy as np
import networkx as nx

def create_frame(dat):
    G = nx.Graph()
    
    #Index(['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate',
    #   'Price', 'CustomerID', 'Country'],
    #  dtype='object')
        
    custData = dat[['CustomerID', 'Country']].drop_duplicates(subset=['CustomerID'], keep='last')
    stockData = dat[['StockCode', 'Description']].drop_duplicates(subset=['StockCode'], keep='last')
    
    assert len(custData) == len(custData.CustomerID.unique())
    assert len(stockData) == len(stockData.StockCode.unique())
    
    for i, stock in stockData.iterrows():
        G.add_node(stock.StockCode,
                   description=stock.Description)
    
    for i, customer in custData.iterrows():
        G.add_node(customer.CustomerID,
                   country=customer.Country)
        
    for i, row in dat.iterrows():
        G.add_edge(row.StockCode,
                   row.CustomerID,
                   key=row.Invoice,
                   quantity=row.Quantity,
                   date=row.InvoiceDate,
                   price=row.Price) # sometimes price changes, sale?
        
    return G
