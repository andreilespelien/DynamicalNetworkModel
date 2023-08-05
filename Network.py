import numpy as np
import matplotlib.pyplot as plt
import math

from Neuron import InputNeuron, IntgrNeuron

class Network():
    tm = 0.1
    dt = 0.00001
    Ia = 1
    tau = 0.005

    def fWeights(self, x):
        return 1.25 * (math.exp(-1/2 * (x / 1.01)**2) - math.exp(-1/2 * (x / 0.202)**2))

    def __init__(self, n, spikeRatio, spatialFrq, pattern):
        self.tVector = np.linspace(0, self.tm, int(self.tm / self.dt), endpoint = False)
        self.n = n
        self.inputNeurons = []
        self.intgrNeurons = []

        for d in range(self.n):
            self.intgrNeurons.append(IntgrNeuron("intgr" + str(d)))
            self.inputNeurons.append(InputNeuron("input" + str(d)))
            self.inputNeurons[d].u[:] = self.Ia
            for i in range(spatialFrq):
                if pattern == 1:
                    self.inputNeurons[d].u[int(self.tm/self.dt * (i * self.n/spatialFrq + d % (self.n/spatialFrq))/self.n) : int(self.tm/self.dt * (i * self.n/spatialFrq + d % (self.n/spatialFrq) + 1)/self.n)] = self.Ia * spikeRatio
                elif pattern == 2:
                    self.inputNeurons[d].u[int(self.tm/self.dt * (i * 2 + d % 2)/(spatialFrq * 2)) : int(self.tm/self.dt * (i * 2 + d % 2 + 1)/(spatialFrq * 2))] = self.Ia * spikeRatio
                elif pattern == 3:
                    self.inputNeurons[d].u[int(self.tm/self.dt * (i * 2 + d // (self.n / 2))/(spatialFrq * 2)) : int(self.tm/self.dt * (i * 2 + d // (self.n / 2) + 1)/(spatialFrq * 2))] = self.Ia * spikeRatio
                
    def connectBasic(self):
        for d in range(self.n):
            self.inputNeurons[d].link(self.intgrNeurons[d], 1)
            self.inputNeurons[d].link(self.intgrNeurons[d - 1], 0.99)
            self.inputNeurons[d].link(self.intgrNeurons[(d + 1) % self.n], 0.99)

            self.intgrNeurons[d].link(self.intgrNeurons[d - 1], 0.99)
            self.intgrNeurons[d].link(self.intgrNeurons[(d + 1) % self.n], 0.99)
            self.intgrNeurons[d].link(self.intgrNeurons[d], 0.01)

    def connectFully(self):
        xWeights = np.linspace(-0.523 * (self.n/2 - 1), 0.523 * self.n/2, self.n)
        yWeights = []
        for x in xWeights:
            yWeights.append(self.fWeights(x))
        yWeights = yWeights[self.n//2 - 1:] + yWeights[:self.n//2 - 1]
        weightsMatrix = []
        
        for d in range(self.n):
            self.inputNeurons[d].link(self.intgrNeurons[d], 1)
            self.inputNeurons[d].link(self.intgrNeurons[d - 1], 0.11)
            self.inputNeurons[d].link(self.intgrNeurons[(d + 1) % self.n], 0.11)

            for d2 in range(self.n):
                self.intgrNeurons[d].link(self.intgrNeurons[d2], yWeights[d2])
            # weightsMatrix.append(yWeights)
            yWeights = [yWeights[-1]] + yWeights[:-1]
        # plt.imshow(weightsMatrix)
        # plt.show()

    def run(self):
        for t in range(len(self.tVector) - 1):
            # print(t)
            for currNeuron in self.intgrNeurons:
                usum = 0
                xsum = 0

                for c in currNeuron.inputs:
                    n = c.n1
                    if type(n) == InputNeuron:
                        usum += n.u[t] * c.w
                    if type(n) == IntgrNeuron:
                        xsum += n.x[t] * c.w

                dx = ((usum - xsum) - currNeuron.x[t]) / self.tau * self.dt
                currNeuron.dx[t + 1] = dx
                currNeuron.x[t + 1] = currNeuron.x[t] + dx

    def plot(self, showDx, n):
        if showDx:
            fig, ax = plt.subplots(n * 3)
            for d in range(n):
                ax[d * 3].plot(self.tVector, self.inputNeurons[d].u)
                ax[d * 3 + 1].plot(self.tVector, self.intgrNeurons[d].x, label = str(d))
                ax[d * 3 + 2].plot(self.tVector, self.intgrNeurons[d].dx)
                ax[d * 3 + 1].legend()
        else:
            fig, ax = plt.subplots(n * 2)
            for d in range(n):
                ax[d * 2].plot(self.tVector, self.inputNeurons[d].u, label = str(d))
                ax[d * 2 + 1].plot(self.tVector, self.intgrNeurons[d].x)
                ax[d * 2].legend()

temp = Network(
    n = 8, 
    spikeRatio = 1.5, 
    spatialFrq = 2, 
    pattern = 1
)
temp.connectFully()
temp.run()
temp.plot(False, 8)
plt.show()