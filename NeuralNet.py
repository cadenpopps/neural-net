import Node
import random
import math
import Constants

class NeuralNet:

    def __init__(self, shape, function):
        self.shape = shape
        self.layers = len(self.shape);

        self.function = function
        self.derivativeOfActivationFunction = Constants.derivativeFunctions[self.function]

        self.nn = self.createNeuralNetwork(self.shape, self.function)

        # print(self)
        # self.trainNeuralNetwork(sampleInputSet, sampleOutputSet, Constants.TRAINING_ITERATIONS)
        # print(self)

    def createNeuralNetwork(self, shape, function):
        nn = []
        numInputs = 1
        numLayer = 0
        for layer in shape:
            nn.append([Node.Node(function, numInputs, numLayer, x) for x in range(layer)])
            numInputs = layer
            numLayer += 1
        return nn

    def trainNeuralNetwork(self, trainingSet):
        trainingSet[1] = self.normalizeOutputs(trainingSet[1])
        print(trainingSet[1])
        for trainingIteration in range(Constants.TRAINING_ITERATIONS):
            for features, expectedOutputs in zip(trainingSet[0], trainingSet[1]):
                predictedOutputs = self.feedForward(features)
                # print("Input(s): " + str(features))
                if(trainingIteration > Constants.TRAINING_ITERATIONS - 2):
                    print("\n\nIteration", trainingIteration, ":")
                    print("predicted output(s): " + str(predictedOutputs[-1]))
                    print("expected output(s): " + str(expectedOutputs))
                self.updateWeights(predictedOutputs, expectedOutputs, Constants.LEARNING_RATE)
        print(self)

    def normalizeOutputs(self, outputs):
        outputRange = 16
        newOutputs = []
        for outputSet in outputs:
            newOutputs.append([])
            for output in outputSet:
                # newOutputs[-1].append(Constants.activationFunctions[self.function](output))
                newOutputs[-1].append(output / outputRange)
        return newOutputs

    def updateWeights(self, predictedOutputs, expectedOutputs, learningRate):
        outputLayerDeltas = self.calculateOutputLayerDeltas(predictedOutputs[-1], expectedOutputs)
        weightedDeltas = self.backwardsPropogate(predictedOutputs, expectedOutputs, outputLayerDeltas)

        # print("output layer error(s): " + str(outputLayerDeltas))
        # print("weighted deltas:" + str(weightedDeltas))
        # print()

        for layer in self.nn:
            for node in layer:
                self.updateNodeWeights(node, predictedOutputs, weightedDeltas, learningRate)

    def updateNodeWeights(self, node, predictedOutputs, weightedDeltas, learningRate):
        wd = weightedDeltas[node.layer][node.index]
        for weightIndex in range(len(node.inputWeights)):
            inp = predictedOutputs[node.layer - 1][weightIndex]
            node.inputWeights[weightIndex] += learningRate * wd * inp
            # print("weighted delta: " + str(wd))
            # print("predicted output: " + str(inp))
            # print("change in weight: " + str(learningRate * wd * inp))
            # print()

        node.inputBias += Constants.BIAS_RATE * wd

    def backwardsPropogate(self, predictedOutputs, expectedOutputs, outputLayerDeltas):
        # deltas = [[] * len(predictedOutputs)]
        deltas = []
        for x in range(len(predictedOutputs)):
            deltas.append([])
        deltas[-1] = outputLayerDeltas.copy()
        for layerIndex in range(self.layers - 2, -1, -1):
            #loop nodes in current layer
            for currentNodeIndex in range(len(predictedOutputs[layerIndex])):
                sumOfWeightedDeltas = 0
                #loop nodes in next layer
                for nextLayerNodeIndex in range(len(deltas[layerIndex + 1])):
                    sumOfWeightedDeltas += deltas[layerIndex + 1][nextLayerNodeIndex] * self.nn[layerIndex + 1][nextLayerNodeIndex].inputWeights[currentNodeIndex]
                deriv = self.derivativeOfActivationFunction(predictedOutputs[layerIndex][currentNodeIndex])
                deltas[layerIndex].append(deriv * sumOfWeightedDeltas)
        return deltas

    def calculateOutputLayerDeltas(self, predictedOutputs, expectedOutputs):
        # outputLayerErr = self.getMSE(predictedOutputs, expectedOutputs)
        outputLayerDeltas = []
        for predictedOutput, expectedOutput in zip(predictedOutputs, expectedOutputs):
            deriv = self.derivativeOfActivationFunction(predictedOutput)
            err = self.getError(predictedOutput, expectedOutput)
            outputLayerDeltas.append(deriv * err)
        return outputLayerDeltas

    def makePrediction(self, inputs):
        outputs = self.feedForward(inputs)
        return max(outputs[-1:])

    def feedForward(self, inputs):
        outputs = [self.feedInputNodes(inputs)]
        for layer in range(1, self.layers):
            outputs.append([])
            for node in self.nn[layer]:
                outputs[layer].append(node.calculateOutput(outputs[layer - 1]))
        return outputs

    def feedInputNodes(self, inputs):
        outputs = []
        for node, inp in zip(self.nn[0], inputs):
            outputs.append(node.calculateOutput(inp))
        return outputs

    def getMSE(self, predictedValues, actualValues):
        err = 0
        for predictedValue, actualValue in zip(predictedValues, actualValues):
            av = Constants.activationFunctions[self.function](actualValue)
            err += math.pow(av - predictedValue, 2)
        return (err/2)

    def getError(self, predictedValue, actualValue):
        return actualValue - predictedValue

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        nnStr = ""
        numLayer = 0
        for layer in self.nn:
            nnStr += "\n==== Layer " + str(numLayer) + " =====\n"
            for node in layer:
                nnStr += "\n" + str(node) + "\n"
            numLayer += 1
            nnStr += "\n"
        return nnStr
