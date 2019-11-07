import networkx as nx
import os


class TemporalGraph:
    
    def __init__(self, node_ordering, index):
        self.frames = {}
        self.index = index # save this as an accessor to which bins exist
        self.node_ordering = node_ordering
    
    def add_frame(self, ind, frame):
        self.frames[ind] = frame
        
    def get_frame(self, ind):
        """ Returns a copy of the frame at the index.
        """
        return self.frames[ind].copy()
        
    def read_gpickles(self, directory):
        for ind_path in os.listdir(directory):
            ind = os.path.basename(ind_path).split(".")[0]
            self.frames[ind] = nx.read_gpickle(ind_path)
            
    def write_gpickles(self, directory):
        for ind, frame in self.frames.items():
            nx.write_gpickle(frame, os.path.join(directory, "%s.pkl" % ind))