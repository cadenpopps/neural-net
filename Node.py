
import random
import math
import Constants

class Node:

    def __init__(self, function, numInputs, layer, index):
        self.function = function
        self.activationFunction = Constants.activationFunctions[self.function]
        self.layer = layer
        self.index = index
        self.numInputs = numInputs
        self.inputWeights = self.initializeRandomWeights(self.numInputs)
        self.inputBias = (random.random() * Constants.INITIAL_BIAS_RANGE)
        if self.layer == 0:
            self.inputWeights = [1]
            self.inputBias = 0 
            self.calculateOutput = self.inputLayerOutput

    def initializeRandomWeights(self, numWeights):
        return [(random.random() * Constants.INITIAL_WEIGHT_RANGE) for x in range(numWeights)]

    def calculateOutput(self, inputs):
        output = self.inputBias
        for i in range(len(inputs)):
            output += inputs[i] * self.inputWeights[i]
        return self.activationFunction(output)

    def inputLayerOutput(self, i):
        return i

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        strNode = "[Node " + str(self.index) + "]\n"
        strNode += "Number of Inputs: " + str(self.numInputs) + "\n"
        strNode += "Input Weights: " + str(self.inputWeights) + "\n"
        strNode += "Input Bias: " + str(self.inputBias)

        return strNode
