import math
import matplotlib.pyplot as plt
import numpy
import backpropagation as net

def plotSin(NN, number, density):
    
    sampleStep = 2 * math.pi / number
    xs = numpy.zeros(density)
    ys = numpy.zeros(density)
    traceStep = 4 * math.pi / density

    for i in range(0,density):
        x = traceStep * i
        NN.setInput(0, (math.sin(x) + 1) / 2)
        NN.setInput(1, (math.sin(x + sampleStep) + 1) / 2)
        NN.apply()
        xs[i] = x
        ys[i] = NN.getOutput(0)

    plt.plot(xs, ys)
    plt.show()