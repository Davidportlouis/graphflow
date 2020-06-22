import numpy as np

class Node(object):
    
    def __init__(self,inbound_nodes=[]):
        self.inbound_nodes = inbound_nodes
        self.outbound_nodes = []
        self.value = None
        for node in inbound_nodes:
            node.outbound_nodes.append(self)

    def  forward(self):
        raise NotImplemented

class Input(Node):
    
    def __init__(self):
        Node.__init__(self)

    def forward(self,value=None):
        if value is not None:
            self.value = value

class Add(Node):

    def __init__(self,x,y):
        Node.__init__(self,[x,y])
    
    def forward(self):
        pass