# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 19:45:11 2018

@author: dirk reichardt
"""
import math
import numpy
import backpropagation as net
import plot_nn as plotter

correctClassifications = 0
last_error = 1000

learned = False
number = 12
iterations = 0
  
inputDim  = 2
hiddenDim = 15
outputDim = 1
  
NN = net.feedForwardNetwork()
NN.configure(inputDim,hiddenDim,outputDim)
NN.init()
NN.setEpsilon(0.001)
NN.setLearningRate(0.3)
  
print ("Generate training dataset:\n")

trainIn = numpy.zeros((number,2))
teach = numpy.zeros((number,1))

step = 2 * math.pi / number
for i in range(0,number):
    x = i * step
    y1 = (math.sin(x) + 1) / 2
    y2 = (math.sin(x+step) + 1) / 2
    y3 = (math.sin(x+step+step) + 1) / 2
    trainIn[i][0] = y1
    trainIn[i][1] = y2
    teach[i][0] = y3
    print("["+str(i)+"] "+ str(y1) + " "+ str(y2) + " -> "+ str(y3))
    
print("\nStarting:\n")

while (correctClassifications < number):
    
    o=numpy.zeros(outputDim)
    t=numpy.zeros(outputDim)
    
    for i in range(0,number):
        iterations=iterations+1
        
        for j in range(0,inputDim):
            NN.setInput(j,trainIn[i][j])
      
        learned = False
        
        single_learn_iterations = 0

        while (not learned):
            single_learn_iterations = single_learn_iterations+1
            NN.apply()

            for j in range(0,outputDim):
                o[j] = NN.getOutput(j)
                t[j] = teach[i][j]

            error = NN.energy(t,o,outputDim);

            if (error > NN.getEpsilon()):
                NN.backpropagate(t)
                                 
            else:
                learned = True
        
        print("backpropagations: "+str(single_learn_iterations))

    # get status of learning

            
    correctClassifications = 0
    total_error = 0
    for i in range(0,number):
        for j in range(0,inputDim):
            NN.setInput(j,trainIn[i][j])
                    
        NN.apply()

        for j in range(0,outputDim):
            o[j] = NN.getOutput(j)
            t[j] = teach[i][j]
        
        error = NN.energy(t,o,outputDim)
        total_error += error

        if (error < NN.getEpsilon()):
            correctClassifications=correctClassifications+1
      
        
    # total error
    last_error = total_error

    print("[" + str(iterations/number) +"] >> Korrekte: " + str(correctClassifications) +" Fehler : "+ str(total_error))

print("Iterationen:"+ str(iterations/number) + "\n")

plotter.plot(NN, step, 30)
raw_input("...")
  
