import numpy as np
import matplotlib.pyplot as plt

from Neuron import InputNeuron, IntgrNeuron
from Connection import Connection

tm = 0.25
dt = 0.0001
tVector = np.arange(0, tm, dt)
Ia = 1
tau = 0.005

intgr1 = IntgrNeuron("intgr1")
intgr2 = IntgrNeuron("intgr2")

input1 = InputNeuron("input1")
input1.u[int(tm * 1/5 / dt) : int(tm * 3/5 / dt)] = Ia
input1.u[int(tm * 3/5 / dt) : int(tm * 5/5 / dt)] = Ia * 1.5
# input1.u[[i for i in range(int(tm * 1/5 / dt), int(tm * 3/5 / dt), 100)]] = 1
# input1.u[[i for i in range(int(tm * 3/5 / dt), int(tm * 5/5 / dt), 80)]] = 1

input2 = InputNeuron("input2")
input2.u[int(tm * 1/5 / dt) : int(tm * 3/5 / dt)] = Ia
input2.u[int(tm * 3/5 / dt) : int(tm * 5/5 / dt)] = Ia * 0.5
# input2.u[[i for i in range(int(tm * 1/5 / dt), int(tm * 3/5 / dt), 100)]] = 1
# input2.u[[i for i in range(int(tm * 3/5 / dt), int(tm * 5/5 / dt), 125)]] = 1


input1.link(intgr1, 1)
input1.link(intgr2, 0.99)
intgr1.link(intgr2, 1.1)
intgr1.link(intgr1, 0.01)
# print([(c.n1.name, c.n2.name) for c in intgr1.inputs])

input2.link(intgr2, 1)
input2.link(intgr1, 0.99)
intgr2.link(intgr1, 1.1)
intgr2.link(intgr2, 0.01)


inputs = [input1, input2]
intgrs = [intgr1, intgr2]

for t in range(len(tVector) - 1):
    for currNeuron in intgrs:
        usum = 0
        xsum = 0

        for c in currNeuron.inputs:
            n = c.n1
            if type(n) == InputNeuron:
                usum += n.u[t] * c.w
            if type(n) == IntgrNeuron:
                xsum += n.x[t] * c.w

        dx = ((usum - xsum) - currNeuron.x[t]) / tau * dt
        currNeuron.dx[t + 1] = dx
        currNeuron.x[t + 1] = currNeuron.x[t] + dx

fig, ax = plt.subplots(6)
ax[0].plot(tVector, input1.u)
ax[1].plot(tVector, intgr1.x)
# ax[2].plot(tVector, intgr1.dx)
ax[3].plot(tVector, input2.u)
ax[4].plot(tVector, intgr2.x)
# ax[5].plot(tVector, intgr2.dx)

# plt.plot(intgr1.x, intgr2.x)
plt.show()