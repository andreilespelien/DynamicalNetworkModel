import numpy as np
import matplotlib.pyplot as plt

from Neuron import InputNeuron, IntgrNeuron
from Connection import Connection

tm = 0.25
dt = 0.0001
tVector = np.arange(0, tm, dt)
Ia = 1
tau = 0.005

ax = plt.figure().add_subplot(projection='3d')

for i in range(1, 6):
    intgr1 = IntgrNeuron("intgr1")
    intgr2 = IntgrNeuron("intgr2")
    intgr3 = IntgrNeuron("intgr3")

    input1 = InputNeuron("input1")
    input1.u[:] = Ia
    input1.u[int(tm * 0/3 / dt) : int(tm * 1/3 / dt)] = Ia * (1 + i * 0.1)

    input2 = InputNeuron("input2")
    input2.u[:] = Ia
    input2.u[int(tm * 1/3 / dt) : int(tm * 2/3 / dt)] = Ia * (1 + i * 0.1)

    input3 = InputNeuron("input3")
    input3.u[:] = Ia
    input3.u[int(tm * 2/3 / dt) : int(tm * 3/3 / dt)] = Ia * (1 + i * 0.1)


    input1.link(intgr1, 1)
    input1.link(intgr3, 0.99)
    input1.link(intgr2, 0.99)
    intgr1.link(intgr3, 1.1)
    intgr1.link(intgr2, 1.1)
    intgr1.link(intgr1, 0.01)

    input2.link(intgr2, 1)
    input2.link(intgr1, 0.99)
    input2.link(intgr3, 0.99)
    intgr2.link(intgr1, 1.1)
    intgr2.link(intgr3, 1.1)
    intgr2.link(intgr2, 0.01)

    input3.link(intgr3, 1)
    input3.link(intgr2, 0.99)
    input3.link(intgr1, 0.99)
    intgr3.link(intgr2, 1.1)
    intgr3.link(intgr1, 1.1)
    intgr3.link(intgr3, 0.01)


    inputs = [input1, input2, input3]
    intgrs = [intgr1, intgr2, intgr3]

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

    # fig, ax = plt.subplots(6)
    # ax[0].plot(tVector, input1.u)
    # ax[1].plot(tVector, intgr1.x)
    # ax[2].plot(tVector, input2.u)
    # ax[3].plot(tVector, intgr2.x)
    # ax[4].plot(tVector, input3.u)
    # ax[5].plot(tVector, intgr3.x)

    ax.plot(intgr1.x, intgr2.x, intgr3.x)
plt.show()