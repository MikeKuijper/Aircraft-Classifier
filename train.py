import machineLearning as magic
from random import shuffle 
import time

# Configuration
classPath = "data/families.txt"                 # Path to the file containing the classifications
networkPath = "data/network_64-families-f.dat"  # Path to the network data file
detailLevel = 2                                 # 1 for manufacturer, 2 for family, 3 for variant

# Customisable variables
imagePath = "data/images_test.txt"              # Path to the file containing image IDs. images_test.txt for testing. images_trainval.txt for training
N = 200                                         # [-] Number of training iterations
verbose = False                                 # Set True for additional logging information
assess = True                                   # Set True for only assessing the network. Enables verbose (override)

continueTraining = False                        # Whether to start from scratch
                                                # Only used if continueTraining is True.

startLearningRate = 1                           # [-] Coefficient determining the size of adjustments. Used if assess is False.
learningRateDecay = 0.9                         # [-] Coefficient of learningRate decay (0-1 typically). Used if assess is False.


# Uncustomisable variables
width = 64      # [px]
height = 32     # [px]
saved = False   # prevents saving multiple times on KeyboardInterrupt
hiddenLayers = [512, 128, 32] # Nodes in the hidden layers

inputMatrices = []
outputMatrices = []

# Load classifications from classPath document and append to list
classifications = []
with open(classPath) as f:
    lines = f.read().split("\n")
    for line in lines:
        if line not in classifications and len(line) > 0:
            classifications.append(line)

# Read imagePath file and get image IDs
with open(imagePath) as f:
    lines = f.read().split("\n")        # Get IDs from file
    lines = list(filter(None, lines))   # Filter out empty lines
    shuffle(lines)                      # Randomly shuffle the lines, and therefore input images

    # Iterate through files with ID in imagePath
    for pictureID in lines:
        if len(pictureID) > 0:          # Filter out empty lines, just in case
            print("\rLoading image {}/{} ({}%) from {}".format(
                lines.index(pictureID) + 1,
                len(lines),
                round((lines.index(pictureID) + 1) / len(lines) * 100),
                imagePath),
                end="")

            # Set up input and output vectors.
            with open("data/images_64/" + pictureID + ".txt") as d:
                dataLines = d.read().split("\n")                # Get the file contents line-by-line. The first line contains the
                                                                # pixel values. The second, third and forth contain the manufacturer,
                                                                # family and variant respectively.
                classification = dataLines[detailLevel]         # Get the desired classification defined by the detailLevel.

                if classification in classifications:           # Filter out any classifications that are to be excluded.
                    inputVector = magic.Vector(width * height)          # Define Vector with elements for every pixel.
                    inputVector.fromList(dataLines[0].split(" "))       # Convert data to Vector object.
                    inputVector.setEach(lambda y, x, val: val/256)      # Normalise all pixel values.
                    inputMatrices.append(inputVector)

                    outputVector = magic.Vector(len(classifications))           # Define Vector with elements for every classification. Filled by default with 0's
                    outputVector[classifications.index(classification)] = 1     # Set the element corresponding to the correct classification to 1
                    outputMatrices.append(outputVector)
    print("")

print("Found {} images in {} matching criteria from {}.".format(len(inputMatrices), imagePath, classPath))
chanceScore = round(100 / len(classifications), 1)  # Expected percentage correct purely by chance [%]


network = magic.NeuralNetwork([width * height] + hiddenLayers + [outputMatrices[0].rows])  # Define Neural Network
if continueTraining or assess:
    network.deserialise(networkPath)
network.learningRate = startLearningRate

if assess:  # If assess mode is enabled, only iterate through images once, and only assess the outcomes without training
    N = 1
    saved = True                # To prevent the network from deserialising the file needlessly
    network.learningRate = 0
    verbose = True              # Enable verbose for manual inspection

try:
    assert len(inputMatrices) == len(outputMatrices)
    for i in range(N):
        network.learningRate *= learningRateDecay
        globalCost = 0
        score = 0
        for j in range(len(inputMatrices)):
            result = network.train(inputMatrices[j], outputMatrices[j])
            globalCost += network.cost

            if max(enumerate(result.toList()), key=lambda x: x[1])[0] == \
                    max(enumerate(outputMatrices[j].toList()), key=lambda x: x[1])[0]:
                score += 1

            if verbose:
                resultList = result.toList()
                sortedIndex = sorted(range(len(resultList)), key=lambda k: -resultList[k])
                correctIndex = max(enumerate(outputMatrices[j].toList()), key=lambda x: x[1])[0]
                maxIndex = max(enumerate(resultList), key=lambda x: x[1])[0]
                confidence = resultList[maxIndex]
                print("\rIteration {}. Image {}/{} ({}%). Cost: {}. Estimated globalCost: {} ({}% / {}%). Output: [{}] {}% {} ({}).".format(
                        i,
                        j + 1,
                        len(inputMatrices),
                        round((j + 1)/len(inputMatrices) * 100, 1),
                        network.cost,
                        round(globalCost / ((j + 1)/len(inputMatrices))),
                        round(100 * (score / ((j + 1)/len(inputMatrices))) / len(inputMatrices), 1),
                        chanceScore,
                        sortedIndex.index(correctIndex) + 1,
                        round(confidence * 100, 1),
                        classifications[maxIndex],
                        classifications[correctIndex]),
                      end="")
            else:
                print("\rIteration {}. Image {}/{} ({}%). Cost: {}.".format(
                        i,
                        j + 1,
                        len(inputMatrices),
                        round((j + 1)/len(inputMatrices) * 100, 1),
                        network.cost),
                      end="")
        print("\rCycle {}. globalCost: {}. Score: {}/{} ({}%)".format(
            i,
            globalCost,
            score,
            len(inputMatrices),
            round(score/len(inputMatrices) * 100, 1)))
        if i != N - 1:  # Don't deserialise if it's the final iteration.

            # Save with a unique filename.
            network.serialise("data/network_64-{}.dat".format(str(int(round(float(str(time.time())), 0)))))
except KeyboardInterrupt:
    print("\nReceived KeyboardInterrupt.")

    # Save with a unique filename.
    network.serialise("data/network_64-{}.dat".format(str(int(round(float(str(time.time())), 0)))))
    saved = True  # Prevent saving again.

if not saved:
    # Save with a unique filename.
    network.serialise("data/network_64-{}.dat".format(str(int(round(float(str(time.time())), 0)))))
