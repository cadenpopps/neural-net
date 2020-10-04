
import random
import math
import Constants

INITIAL_WEIGHT_RANGE = 4
INITIAL_BIAS_RANGE = .2

class Node:

    def __init__(self, function, numInputs, layer):

        self.function = function
        self.activationFunction = Constants.activationFunctions[self.function]
        # self.derivativeFunction = Constants.derivativeFunctions[self.function]

        self.numInputs = numInputs
        self.layer = layer
        self.inputWeights = self.initializeRandomWeights(self.numInputs)
        self.inputBias = (random.random() * INITIAL_BIAS_RANGE) - (INITIAL_BIAS_RANGE / 2)

        if self.layer == 0:
            self.calculateOutput = self.inputLayerCalculateOutput

    def initializeRandomWeights(self, numWeights):
        return [((random.random() * INITIAL_WEIGHT_RANGE) - (INITIAL_WEIGHT_RANGE / 2)) for x in range(numWeights)]

    # def initializeRandomBiases(self, numBiases):
    #     return [(random.random() * INITIAL_BIAS_RANGE) for x in range(numBiases)]

    def calculateOutput(self, inputs):
        output = self.inputBias
        for i in range(len(inputs)):
            output += inputs[i] * self.inputWeights[i]
        return self.activationFunction(output)

    def inputLayerCalculateOutput(self, singleInput):
        return self.activationFunction(self.inputBias + (singleInput * self.inputWeights[0]))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        strNode = "[Node]\n"
        strNode += "Number of Inputs: " + str(self.numInputs) + "\n"
        strNode += "Input Weights: " + str(self.inputWeights) + "\n"
        strNode += "Input Bias: " + str(self.inputBias)

        return strNode
