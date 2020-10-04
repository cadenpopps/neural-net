import Node
import random
import Constants

class NeuralNet:

    def __init__(self, shape, function):
        self.shape = shape
        self.layers = len(self.shape);

        self.function = function
        self.derivativeOfActivationFunction = Constants.derivativeFunctions[self.function]

        self.nn = self.createNeuralNetwork(self.shape, self.function)

        sampleInputSet = []
        sampleOutputSet = []
        numSampleInputFeatures = self.shape[0]
        numSampleOutputs = self.shape[-1]
        elementsInSampleSet = 10
        for x in range(elementsInSampleSet):
            sampleOutputSet.append([])
            for output in range(numSampleOutputs):
                sampleOutputSet[x].append(x)
            sampleInputSet.append([])
            for feature in range(numSampleInputFeatures):
                if (feature % 2 == 0):
                    sampleInputSet[x].append(x - 1)
                else:
                    sampleInputSet[x].append(x + 1)

        self.trainNeuralNetwork(sampleInputSet, sampleOutputSet, Constants.TRAINING_ITERATIONS)
        print(self)

    def createNeuralNetwork(self, shape, function):
        nn = []
        numInputs = 1
        numLayer = 0
        for layer in shape:
            nn.append([Node.Node(function, numInputs, numLayer, x) for x in range(layer)])
            numInputs = layer
            numLayer += 1
        return nn

    def trainNeuralNetwork(self, inputSet, outputSet, iterations):
        for trainingIteration in range(iterations):
            print("\nIteration", trainingIteration, ":")
            for features, expectedOutputs in zip(inputSet, outputSet):
                predictedOutputs = self.feedForward(features)
                self.updateWeights(predictedOutputs, expectedOutputs, Constants.LEARNING_RATE)


    def updateWeights(self, predictedOutputs, expectedOutputs, learningRate):
        outputLayerDeltas = self.calculateOutputLayerDeltas(predictedOutputs[-1], expectedOutputs)
        weightedDeltas = self.backwardsPropogate(predictedOutputs, expectedOutputs, outputLayerDeltas)

        print("predicted output(s): " + str(predictedOutputs[-1]))
        print("expected output(s): " + str(expectedOutputs))
        print()
        # print("output layer error(s): " + str(outputLayerDeltas))

        for layer in self.nn[1:]:
            for node in layer:
                self.updateNodeWeights(node, predictedOutputs, weightedDeltas, learningRate)

    def updateNodeWeights(self, node, predictedOutputs, weightedDeltas, learningRate):
        wd = weightedDeltas[node.layer][node.index]
        for weightIndex in range(len(node.inputWeights)):
            po = predictedOutputs[node.layer - 1][weightIndex]
            print("weighted delta: " + str(wd))
            print("predicted output: " + str(po))
            print("change in weight: " + str(learningRate * wd * po))
            node.inputWeights[weightIndex] -= learningRate * wd * po
        node.inputBias -= Constants.BIAS_RATE * weightedDeltas[node.layer][node.index]

    def backwardsPropogate(self, predictedOutputs, expectedOutputs, outputLayerDeltas):
        deltas = predictedOutputs.copy()
        deltas[-1] = outputLayerDeltas
        for layerIndex in range(self.layers - 2, -1, -1):
            #loop nodes in current layer
            for currentNodeIndex in range(len(predictedOutputs[layerIndex])):
                deriv = self.derivativeOfActivationFunction(predictedOutputs[layerIndex][currentNodeIndex])
                sumOfWeightedDeltas = 0
                #loop nodes in next layer
                for nextLayerNodeIndex in range(len(deltas[layerIndex + 1])):
                    sumOfWeightedDeltas += deltas[layerIndex + 1][nextLayerNodeIndex] * self.nn[layerIndex + 1][nextLayerNodeIndex].inputWeights[currentNodeIndex]
                deltas[layerIndex][currentNodeIndex] = (deriv * sumOfWeightedDeltas)

        return deltas

    def calculateOutputLayerDeltas(self, predictedOutputs, expectedOutputs):
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

    def getError(self, predictedValue, actualValue):
        return (actualValue - predictedValue)

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
