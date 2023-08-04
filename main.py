import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy.matlib as npm
import numpy.linalg as npl
import pprint

from Network import Network

def PCA(data):
    mean = np.mean(data)
    means = npm.repmat(mean, len(data), 1)
    norm = data - means

    cov = (norm @ norm.T) / len(data)
    (d, v) = npl.eig(cov)
    idx = d.argsort()[::-1]
    dsort = d[idx]
    vsort = v[:,idx]
    
    varExplained = np.cumsum(dsort)/sum(dsort)
    normProj = vsort.T[:3, :] @ norm
    return normProj 

# ax = plt.figure().add_subplot()
ax = plt.figure().add_subplot(projection='3d')

# for i in [1.2, 1.3]:
for i in np.linspace(1.0, 1.5, 30):
    network = Network(3, i)
    network.run()

    neurons = np.array([neuron.x for neuron in network.intgrNeurons])
    reduced = PCA(neurons)
    ax.plot(
        reduced[0],
        reduced[1], 
        reduced[2]
        # color = (148/255, 0/255, 211/255, (i + 1) / 101 + 0.25)
    )

    # dervtvs = np.array([neuron.dx for neuron in network.intgrNeurons])
    # reduced = PCA(dervtvs)
    # ax.plot(
    #     reduced[0],
    #     reduced[1], 
    #     reduced[2]
    #     # color = (148/255, 0/255, 211/255, (i + 1) / 101 + 0.25)
    # )

plt.show()