
import Node
import random
import Constants

class NeuralNetwork:

    def __init__(self, shape, function):
        self.shape = shape
        self.layers = len(self.shape);

        self.function = function
        self.derivativeOfActivationFunction = Constants.derivativeFunctions[self.function]

        self.nn = self.createNeuralNetwork(self.shape, self.function)

        sampleInputSet = []
        sampleOutputSet = []
        elementsInSampleSet = 5
        for x in range(elementsInSampleSet):
            sampleOutputSet.append([])
            for output in range(self.shape[-1]):
                sampleOutputSet[x].append(random.random() * x)
            sampleInputSet.append([])
            for feature in range(self.shape[0]):
                sampleInputSet[x].append(10 * random.random())

        self.trainNeuralNetwork(sampleInputSet, sampleOutputSet, 3)

        # self.makePrediction([0, 5, 10])
        # print("")
        # self.makePrediction([0, -5, -10])
        # print("")
        # self.makePrediction([0, 0, 0])

    def createNeuralNetwork(self, shape, function):
        nn = []
        numInputs = 1
        numLayer = 0
        for layer in shape:
            nn.append([Node.Node(function, numInputs, numLayer) for x in range(layer)])
            numInputs = layer
            numLayer += 1
        return nn

    def trainNeuralNetwork(self, inputSet, outputSet, iterations):
        for trainingIteration in range(iterations):
            print("\nIteration", trainingIteration, ":")
            for features, expectedOutputs in zip(inputSet, outputSet):
                outputs = self.feedForward(features)
                outputLayerErrors = self.calculateOutputLayerErrors(outputs[-1], expectedOutputs)
                # print(outputs)
                print("predicted output(s): " + str(outputs[-1]))
                print("expected output(s): " + str(expectedOutputs))
                print("output layer error(s): " + str(outputLayerErrors))
                print()
                # pred = self.makePrediction(features)
                # print(pred)

    def calculateOutputLayerErrors(self, predictedOutputs, expectedOutputs):
        outputLayerErrors = []
        for predictedOutput, expectedOutput in zip(predictedOutputs, expectedOutputs):
            outputLayerErrors.append(self.derivativeOfActivationFunction(predictedOutput) * (expectedOutput - predictedOutput))
        return outputLayerErrors

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
