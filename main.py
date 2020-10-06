import NeuralNet
import Constants
import pandas as pd

DEFAULT_TRAINING_DATASET = "http://cadenpopps.com/machine-learning/training.data"

def Main():

    datafile = safeLoadDataset(DEFAULT_TRAINING_DATASET)
    trainingDataset = []
    trainingDataset.append(datafile.values[3:,:-1])
    trainingDataset.append(datafile.values[3:,-1:])

    inputs = trainingDataset[0].shape[1]

    neuralnet = NeuralNet.NeuralNet([inputs, 6, 1], Constants.SIGMOID)
    neuralnet.trainNeuralNetwork(trainingDataset)



def safeLoadDataset(filename):
    try:
        print("Trying to load dataset:", filename)
        data = pd.read_csv(filename)
        print("Successfully loaded dataset:", filename, "\n")
        return data
    except FileNotFoundError:
        print("File", filename, " not found, exiting.")
        return

Main()
