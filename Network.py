import numpy as np
import matplotlib.pyplot as plt
import math

from Neuron import InputNeuron, IntgrNeuron

class Network():
    tm = 0.1
    dt = 0.00001
    tVector = np.arange(0, tm, dt)
    Ia = 1
    tau = 0.005

    def fWeights(self, x):
        return 2.5 * (math.exp(-1/2 * (x / 1.01)**2) - math.exp(-1/2 * (x / 0.202)**2))

    def __init__(self, n, spikeRatio):
        self.n = n
        self.inputNeurons = []
        self.intgrNeurons = []

        for d in range(n):
            self.intgrNeurons.append(IntgrNeuron("intgr" + str(d)))
            self.inputNeurons.append(InputNeuron("input" + str(d)))
            self.inputNeurons[d].u[:] = self.Ia
            self.inputNeurons[d].u[int(self.tm * d/n / self.dt) : int(self.tm * (d+1)/n / self.dt)] = self.Ia * spikeRatio

        xWeights = np.linspace(-4, 4, n)
        yWeights = []
        for x in xWeights:
            yWeights.append(self.fWeights(x))
        # print(yWeights)
        # plt.plot(xWeights, yWeights)
        # plt.show()
        
        for d in range(n):
            self.inputNeurons[d].link(self.intgrNeurons[d], 1)
            self.inputNeurons[d].link(self.intgrNeurons[d - 1], 0.99)
            self.inputNeurons[d].link(self.intgrNeurons[(d + 1) % n], 0.99)

            self.intgrNeurons[d].link(self.intgrNeurons[d - 1], 0.99)
            self.intgrNeurons[d].link(self.intgrNeurons[(d + 1) % n], 0.99)
            self.intgrNeurons[d].link(self.intgrNeurons[d], 0.01)
        # print([c.n1.name for c in self.intgrNeurons[0].inputs])

    def run(self):
        for t in range(len(self.tVector) - 1):
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

    def plot(self, showDx):
        if showDx:
            fig, ax = plt.subplots(self.n * 3)
            for d in range(self.n):
                ax[d * 3].plot(self.tVector, self.inputNeurons[d].u)
                ax[d * 3 + 1].plot(self.tVector, self.intgrNeurons[d].x, label = str(d))
                ax[d * 3 + 2].plot(self.tVector, self.intgrNeurons[d].dx)
                ax[d * 3 + 1].legend()
        else:
            fig, ax = plt.subplots(self.n * 2)
            for d in range(self.n):
                print(d)
                ax[d * 2].plot(self.tVector, self.inputNeurons[d].u, label = str(d))
                ax[d * 2 + 1].plot(self.tVector, self.intgrNeurons[d].x)
                ax[d * 2].legend()

# temp = Network(32, 1.5)