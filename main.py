import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy.matlib as npm
import numpy.linalg as npl
import pprint

from Network import Network

# ax = plt.figure().add_subplot()
ax = plt.figure().add_subplot(projection='3d')

# for i in range(21):
#     network = Network(3, 1 + i / 20)
#     network.run()
#     ax.plot(network.intgrNeurons[0].x, network.intgrNeurons[1].x, network.intgrNeurons[2].x)

for i in range(11):
    network = Network(64, 1 + i / 20)
    network.run()
    neurons = np.array([neuron.x for neuron in network.intgrNeurons])
    # print(neurons)

    mean = np.mean(neurons)
    means = npm.repmat(mean, len(neurons), 1)
    norm = neurons - means
    # print(means)
    # print(norm.shape)

    cov = (norm @ norm.T) / len(neurons)
    # print(cov.shape)
    # im = plt.imshow(cov)
    # plt.show()

    (d, v) = npl.eig(cov)
    idx = d.argsort()[::-1]
    dsort = d[idx]
    vsort = v[:,idx]
    # print(idx)
    
    varExplained = np.cumsum(dsort)/sum(dsort)
    normProj = vsort.T[:3, :] @ norm
    # print(normProj.shape)

    # for i in range(len(normProj[0])):
    #     x = normProj[0][i]
    #     y = normProj[1][i]
    #     z = normProj[2][i]
    #     ax.plot(x, y, z, marker = ",")
    threshold = int(network.tm / network.dt)
    ax.plot(
        normProj[0][:threshold], 
        normProj[1][:threshold], 
        normProj[2][:threshold] 
        # color = (148/255, 0/255, 211/255, (i + 1) / 101 + 0.25)
    )
ax.legend()

plt.set_cmap("viridis")
plt.show()