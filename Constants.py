import math

SIGMOID = 0
TANH = 1
RELU = 2

BIAS_RATE = .01
LEARNING_RATE = .05
TRAINING_ITERATIONS = 1000

INITIAL_WEIGHT_RANGE = .2
INITIAL_BIAS_RANGE = .2


def sigmoidTransformation(x):
    return (1 / (1 + math.exp(-x)))

def tanhTransformation(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def reluTransformation(x):
    return max(0, x)


# def sigmoidDerivative(x):
#     return sigmoidTransformation(x) * (1 - sigmoidTransformation(x))

def sigmoidDerivative(x):
    return x * (1 - x)

def tanhDerivative(x):
    return 1 - math.pow(x, 2)

def reluDerivative(x):
    if x >= 0: return 1
    else: return 0

activationFunctions = []
activationFunctions.append(sigmoidTransformation)
activationFunctions.append(tanhTransformation)
activationFunctions.append(reluTransformation)

derivativeFunctions = []
derivativeFunctions.append(sigmoidDerivative)
derivativeFunctions.append(tanhDerivative)
derivativeFunctions.append(reluDerivative)

