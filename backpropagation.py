# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 11:49:28 2018

@author: dirk reichardt

"""

import numpy.ma

from math import exp
from random import random


class feedForwardNetwork():

    def __init__ (self):
    
        self.MAX_INPUT_LAYER_SIZE=20
        self.MAX_HIDDEN_LAYER_SIZE=40
        self.MAX_OUTPUT_LAYER_SIZE=20

        self.INPUT_TO_HIDDEN=0
        self.HIDDEN_TO_OUTPUT=1  

        self.DEFAULT_EPSILON=1
        self.DEFAULT_LEARNING_RATE=0.5    
    
        self.inNeurons = 0
        self.hiddenNeurons = 0
        self.outNeurons = 0
    
        self.InputLayer = numpy.zeros(self.MAX_INPUT_LAYER_SIZE)
        self.HiddenLayer = numpy.zeros(self.MAX_HIDDEN_LAYER_SIZE)
        self.OutputLayer = numpy.zeros(self.MAX_OUTPUT_LAYER_SIZE)
        
        self.weightsToHidden = numpy.zeros((self.MAX_INPUT_LAYER_SIZE+1,self.MAX_HIDDEN_LAYER_SIZE))
        self.weightsToOutput = numpy.zeros((self.MAX_HIDDEN_LAYER_SIZE+1,self.MAX_OUTPUT_LAYER_SIZE))
        
        
    def configure(self,inLayer, hidden, out):
        if ((inLayer > 0) and (inLayer < self.MAX_INPUT_LAYER_SIZE)):
            self.inNeurons = inLayer
        else:
            self.inNeurons = 1
            
        if ((hidden > 0) and (hidden < self.MAX_HIDDEN_LAYER_SIZE)):
            self.hiddenNeurons = hidden
        else:
            self.hiddenNeurons = 1
            
        if ((out > 0) and (out < self.MAX_OUTPUT_LAYER_SIZE)):
            self.outNeurons = out
        else:
            self.outNeurons = 1

        self.epsilon = self.DEFAULT_EPSILON
        self.learningRate = self.DEFAULT_LEARNING_RATE

    def init(self):    # initialize weights
              
        # all neuron activations set to 0

        for i in range(0,self.MAX_INPUT_LAYER_SIZE):
            self.InputLayer[i] = 0

        self.InputLayer[self.inNeurons] = 1  # threshold activation (common trick)

        for i in range(0,self.MAX_HIDDEN_LAYER_SIZE):
            self.HiddenLayer[i] = 0

        self.HiddenLayer[self.hiddenNeurons] = 1  # threshold activation (common trick)
        
        for i in range(0,self.MAX_OUTPUT_LAYER_SIZE):
            self.OutputLayer[i] = 0
          
        # all weights are set to 0
      
        for i in range(0,self.MAX_INPUT_LAYER_SIZE+1):
          for j in range(0,self.MAX_HIDDEN_LAYER_SIZE):
              self.weightsToHidden[i][j] = 0

        for i in range(0,self.MAX_HIDDEN_LAYER_SIZE+1):
          for j in range(0,self.MAX_OUTPUT_LAYER_SIZE):
              self.weightsToOutput[i][j] = 0

        # the weights of the configured net (node subset)
        # are set to a random number between -0.5 and 0.5

        for i in range(0,self.MAX_INPUT_LAYER_SIZE+1):
          for j in range(0,self.MAX_HIDDEN_LAYER_SIZE):
              self.weightsToHidden[i][j] = ((random()*100-50)/100)
              

        for i in range(0,self.MAX_HIDDEN_LAYER_SIZE+1):
          for j in range(0,self.MAX_OUTPUT_LAYER_SIZE):
              self.weightsToOutput[i][j] = ((random()*100-50)/100)

    @staticmethod
    def t(x):   # sigmoid function
        return (1/(1 + exp(-x)))

    def setInput(self,x,value):
        if ((x >= 0) and (x < self.inNeurons) and (value >= 0) and (value <= 1)):
            self.InputLayer[x] = value

    def setOutput(self,x,value):
        if ((x >= 0) and (x < self.outNeurons) and (value >= 0) and (value <= 1)):
            self.OutputLayer[x] = value

    def getInput(self,x):
        ret = -1
        if ((x >= 0) and (x < self.inNeurons)):
            ret = self.InputLayer[x]
        return ret

    def getOutput(self,x):
        ret = -1
        if ((x >= 0) and (x < self.outNeurons)):
            ret = self.OutputLayer[x]
        return ret

    def getHidden(self,x):
        ret = -1
        if ((x >= 0) and (x < self.hiddenNeurons)):
            ret = self.HiddenLayer[x]
        return ret

    def getWeight(self,layer,x,y):
        ret = -1
        if (layer == self.INPUT_TO_HIDDEN): # from input to hidden       
            if ((x >= 0) and (x < self.inNeurons+1) and   #includes threshold
                (y >= 0) and (y < self.hiddenNeurons)):
                    ret = self.weightsToHidden[x][y]
                
        if (layer == self.HIDDEN_TO_OUTPUT): # from hidden layer to output
            if ((x >= 0) and (x < self.hiddenNeurons+1) and   #includes threshold
                (y >= 0) and (y < self.outNeurons)):
                    ret = self.weightsToOutput[x][y]
        return ret
    
    def apply(self):
        # propagate activation through the net
        # compute hidden layer activation

        self.InputLayer[self.inNeurons]= 1  # for threshold computation

        for j in range(0,self.hiddenNeurons):
            net = 0 # netto input of a neuron
                
            for i in range(0,self.inNeurons+1):          
                net += self.weightsToHidden[i][j]*self.InputLayer[i]
                self.HiddenLayer[j] = self.t(net)  # using transfer function (sigmoid)
        

        for j in range(0,self.outNeurons):
            net = 0 # netto input of a neuron

            for i in range(0,self.hiddenNeurons+1):
                net += self.weightsToOutput[i][j]*self.HiddenLayer[i]
                self.OutputLayer[j] = self.t(net)  # using transfer function (sigmoid)
                
                
    def backpropagate(self,t):
        
        # neural network learning step

        deltaH = numpy.zeros(self.hiddenNeurons+1)
        
        e = self.energy(t,self.OutputLayer,self.outNeurons)

        if (self.epsilon < e):
        
            # backpropagation
            # update weights to output layer
            # Formula :  delta_wij = lernrate dj hiddenlayer_i
            #                   dj = (tj-yj)yj(1-yj)
                      
            for j in range(0,self.outNeurons):
                y = self.OutputLayer[j]
                delta = (t[j]-y)*y*(1-y)

                for i in range(0,self.hiddenNeurons+1):
                    deltaH[i] += delta * self.weightsToOutput[i][j]
                    self.weightsToOutput[i][j] += self.learningRate * delta * self.HiddenLayer[i]

            for i in range(0,self.hiddenNeurons):
                delta = deltaH[i]*self.HiddenLayer[i]*(1-self.HiddenLayer[i])
                
                for j in range(0,self.inNeurons+1):
                    self.weightsToHidden[j][i] += self.learningRate * delta * self.InputLayer[j]
                    
            

    def getEpsilon(self):
        return self.epsilon

    def getLearningRate(self):
        return self.learningRate

    def setEpsilon(self,eps):
        if (eps > 0):
            self.epsilon = eps

    def setLearningRate(self,mu):
        if ((mu > 0) and (mu <= 10)):
            self.learningRate = mu
    
    @staticmethod
    def energy(t,y,num):
        energy = 0
        
        for i in range(0,num):
            energy += (t[i]-y[i])*(t[i]-y[i])
   
        energy /= 2

        return energy

    def setInputLayer(self,inputArray):
        self.InputLayer = inputArray
   
    def setWeights(self,w1,w2):
        
        for i in range(0,self.inNeurons+1):
           for j in range(0,self.hiddenNeurons):
               self.weightsToHidden[i][j] = w1[i][j]

        for i in range(0,self.hiddenNeurons+1):
            for j in range(0,self.outNeurons):
                self.weightsToOutput[i][j] = w2[i][j]

    def getWeights(self,w1,w2):
        for i in range(0,self.inNeurons+1):
            for j in range(0,self.hiddenNeurons):
                w1[i][j] = self.weightsToHidden[i][j] 

        for i in range(0,self.hiddenNeurons+1):
            for j in range(0,self.outNeurons):
                w2[i][j] = self.weightsToOutput[i][j]

    def setWeight(self,level,i,j,w):
        if (not ((level > 1) or (level < 0))):
            if (level == 0):
                if ((i >= 0) and (i < self.inNeurons+1) and
                    (j >= 0) and (j < self.hiddenNeurons)):
                    self.weightsToHidden[i][j] = w
            if (level == 1):
                if ((i >= 0) and (i < self.hiddenNeurons+1) and
                    (j >= 0) and (j < self.outNeurons)):
                    self.weightsToOutput[i][j] = w
    
    