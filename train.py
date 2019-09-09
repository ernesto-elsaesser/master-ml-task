import math
import numpy
import backpropagation as net

def train(trainIn, teach, number):
    
    correctClassifications = 0

    inputDim  = 21
    hiddenDim = 50
    outputDim = 3

    NN = net.feedForwardNetwork()
    NN.configure(inputDim,hiddenDim,outputDim)
    NN.init()
    NN.setEpsilon(0.001)
    NN.setLearningRate(0.3)
        
    print("\nStarting:\n")

    learned = False
    iterations = 0

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

        print("[" + str(iterations/number) +"] >> Korrekte: " + str(correctClassifications) +" Fehler : "+ str(total_error))

    print("Iterationen:"+ str(iterations/number) + "\n")
    return(NN)
