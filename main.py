import numpy as np
import matplotlib.pyplot as plt
import math
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

def normalize(vectors):
    for vector in vectors:
        mag = 0
        for dim in vector:
            mag += dim ** 2
        mag = math.sqrt(mag)
        vector /= mag
    return vectors

ax1 = plt.figure().add_subplot(projection='3d')
ax2 = plt.figure().add_subplot()
ax3 = plt.figure().add_subplot(projection='3d')
ax4 = plt.figure().add_subplot()
ax5 = plt.figure().add_subplot()
endpoints = []

# for i in [1.5]:
for i in np.linspace(1.0, 3.0, 30):
    if i == 1.0:
        continue

    # 3, 16, 32, 64
    network = Network(16, i) # 6 IS INTERESTING FOR FULLY CONNECTED
    # network.connectBasic()
    network.connectFully()
    network.run()
    print(i)

    neurons = np.array([neuron.x for neuron in network.intgrNeurons])
    endpoints.append(neurons[:, -1])

    reduced1 = PCA(neurons)
    ax1.plot(
        reduced1[0],
        reduced1[1], 
        reduced1[2],
        # label = i,
        color = (1 - (i - 1) / 2, 0, (i - 1) / 2) # 148/255, 0/255, 211/255
    )
    ax2.plot(
        reduced1[0],
        reduced1[1],
        color = (1 - (i - 1) / 2, 0, (i - 1) / 2)
    )

    dervtvs = np.array([neuron.dx for neuron in network.intgrNeurons])
    reduced2 = PCA(dervtvs)
    ax3.plot(
        reduced2[0],
        reduced2[1], 
        reduced2[2],
        color = (1 - (i - 1) / 2, 0, (i - 1) / 2)
    )
    ax4.plot(
        reduced2[0],
        reduced2[1],
        color = (1 - (i - 1) / 2, 0, (i - 1) / 2)
    )

print("----------------------")
endpoints = np.array(endpoints)
# for i in range(len(endpoints)):
#     ax1.plot(endpoints[i][0], endpoints[i][1], endpoints[i][2], marker = ".", color = (1, 0, 0, (i + 1) / len(endpoints)))

gapMatrix = []
for i in range(len(endpoints) - 1):
    gapRow = []
    for j in range(len(endpoints) - 1):
        gapRow.append((endpoints[i + 1] - endpoints[i])[:3])
    gapMatrix.append(normalize(gapRow))
ax5.imshow(gapMatrix)

plt.show()