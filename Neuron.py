import numpy as np

from Connection import Connection

class Neuron():
    tm = 0.25
    dt = 0.0001

    def __init__(self):
        pass

    def link(self, n2, w):
        newc = Connection(self, n2, w)
        self.outputs.append(newc)
        n2.inputs.append(newc)
    
    def getIn(self, n1):
        for c in self.inputs:
            if c.n1 == n1:
                return c
    def getOut(self, n2):
        for c in self.outputs:
            if c.n2 == n2:
                return c

class InputNeuron(Neuron):
    def __init__(self, name):
        self.inputs = []
        self.outputs = []

        self.name = name
        self.u = np.zeros(int(self.tm / self.dt))

class IntgrNeuron(Neuron):
    def __init__(self, name):
        self.inputs = []
        self.outputs = []
        
        self.name = name
        self.x = np.zeros(int(self.tm / self.dt))
        self.dx = np.zeros(int(self.tm / self.dt))