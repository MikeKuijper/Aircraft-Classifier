import copy
import math
import random
import numpy as np


# machineLearning.py contains the necessary class definitions for the implementation of (deep) neural networks.

def sigmoid(x):
    """Returns the normalised value of a given x. On overflow, it rounds to the function's limits of either 0 or 1"""
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        if x >= 0:
            return 1
        else:
            return 0


def sigmoidDerivative(x):
    """Returns the derivative of the sigmoid function"""
    return x * (1 - x)


def matrixMult(a, b):  # self * other
    """Multiplies two Matrix objects with one another. Applies a to b if the matrices are compatible"""
    assert a.cols == b.rows
    res = Matrix(a.rows, b.cols)

    # Convert to numpy arrays because it's much faster
    a_array = np.asarray(a.data)
    b_array = np.asarray(b.data)

    # Using numpy matrix multiplication
    c = np.matmul(a_array, b_array)

    # Converting back to custom Matrix class
    res.data = c.tolist()
    return res

    # For the curious minds:
    # for y in range(res.rows):
    #     for x in range(res.cols):
    #         localSum = 0
    #         for i in range(a.cols):
    #             localSum += a[y, i] * b[i, x]
    #         res[y, x] = localSum
    # return res


def scalarMult(m, s):
    """Returns the product of a matrix and a scalar"""
    res = Matrix(m.rows, m.cols)
    for y in range(res.rows):
        for x in range(res.cols):
            res[y, x] = m[y, x] * s
    return res


class Matrix:
    """Matrix class for the computations with the weights and biases.
    It creates an m by n matrix and takes positional arguments in the
    form of [row, column]"""

    def __init__(self, _rows, _cols):
        self.rows = _rows
        self.cols = _cols
        self.size = _rows * _cols
        self.data = [[0 for x in range(_cols)] for y in range(_rows)]

    def __str__(self):
        """Handling str() for serialisation"""
        out = "["
        for y in range(self.rows):
            out += "["
            for x in range(self.cols):
                out += str(self.data[y][x])
                if x == self.cols - 1:
                    if y == self.rows - 1:
                        out += "]"
                    else:
                        out += "];"
                else:
                    out += ","
        out += "]"
        return out

    def __setitem__(self, _key, _value):
        """Handling Matrix[a, b] = c"""
        # if isinstance(key, int):
        #    self.data[key][0] = value
        # else:
        self.data[_key[0]][_key[1]] = _value

    def __getitem__(self, _key):
        """Handling Matrix[a, b]"""
        if isinstance(_key, int):
            return self.data[_key][0]
        else:
            return self.data[_key[0]][_key[1]]

    def __mul__(self, _other):
        """Handling Matrix A, B: A*B"""
        if type(_other) == float or type(_other) == int:
            return scalarMult(self, _other)
        if isinstance(_other, Matrix):
            return matrixMult(self, _other)

    def __rmul__(self, _other):
        """Handling Matrix A, B: B*A"""
        if type(_other) == float or type(_other) == int:
            return scalarMult(self, _other)
        if isinstance(_other, Matrix):
            return matrixMult(_other, self)

    def __add__(self, _other):
        """Handling Matrix A, B: A+B"""
        assert isinstance(_other, Matrix)                             # Assert that both are matrices.
        assert _other.rows == self.rows and _other.cols == self.cols  # Assert that matrices have matching dimensions
        res = Matrix(self.rows, self.cols)  # Define new Matrix with either of their dimensions
        for y in range(self.rows):
            for x in range(self.cols):
                res[y, x] = self[y, x] + _other[y, x] # Sum all indices
        return res

    def __rsub__(self, _other):
        """Handling Matrix A, B: A-B"""
        return _other + -1 * self

    def __sub__(self, _other):
        """Handling Matrix A, B: B-A"""
        return self + -1 * _other

    def fill(self, _val):
        """Filling the matrix with a (numerical) value"""
        for y in range(self.rows):
            for x in range(self.cols):
                self.data[y][x] = _val      # Set each element equal to _val
        return self

    def print(self):
        """Printing the matrix contents. For debugging purposes only."""
        for y in range(self.rows):
            if y == 0:
                print("[", end="")
            else:
                print(" ", end="")
            for x in range(self.cols):
                if x == self.cols - 1:
                    print(self.data[y][x], end="")
                else:
                    print(self.data[y][x], end=" ")
            if y == self.rows - 1:
                print("]")
            else:
                print("\n", end="")

    def __copy__(self):
        """Handling copy"""
        res = Matrix(self.rows, self.cols)
        res.data = copy.deepcopy(self.data)
        return res

    def __deepcopy__(self, _memodict):
        """Handling deepcopy"""
        return self.__copy__()

    def forEach(self, _f):
        """Execute function for each element in the matrix"""
        for y in range(self.rows):
            for x in range(self.cols):
                _f(y, x, self.data[y][x])

    def setEach(self, _f):
        """Set each element in the matrix according to function. Arguments passed: y, x, value"""
        for y in range(self.rows):
            for x in range(self.cols):
                self.data[y][x] = _f(y, x, self.data[y][x])

    def randomise(self, _lower=-1, _upper=1):
        """Randomly assigns value to each matrix element"""
        self.setEach(lambda y, x, v: random.uniform(_lower, _upper))

    def toList(self):
        """Returns a 1D list of each element in the matrix. Horizontal then vertical."""
        output = []
        self.forEach(lambda y, x, val: output.append(val))
        return output


class Vector(Matrix):
    """Horizontal, single column Matrix"""

    def __init__(self, _rows):
        if isinstance(_rows, list):
            Matrix.__init__(self, len(_rows), 1)
            self.fromList(_rows)
        else:
            Matrix.__init__(self, _rows, 1)

    def __setitem__(self, _key, _value):
        self.data[_key][0] = _value

    def fromList(self, _list):
        assert self.rows == len(_list)
        self.setEach(lambda y, x, val: float(_list[y]))
        return self

    def toList(self):
        output = []
        self.forEach(lambda y, x, val: output.append(val))
        return output


class NeuralNetwork:
    def __init__(self, _layersOrPath=None):
        """Neural network class constructor."""

        if _layersOrPath is None:
            _layersOrPath = []
        if isinstance(_layersOrPath, list):
            self.structure = _layersOrPath
            self.length = len(_layersOrPath)
        else:
            self.structure = []
            self.length = 0
        self.learningRate = 1
        self.globalError = 0
        self.cost = 0

        self.weightMatrices = []
        self.biasMatrices = []
        self.errorMatrices = []
        self.layerMatrices = []

        if isinstance(_layersOrPath, str):
            self.deserialise(_layersOrPath)

        if _layersOrPath is not None:
            self.initNetwork()

    def initNetwork(self):
        """Initialises the network structure, defining the necessary matrices based on self.structure"""

        self.weightMatrices.clear()
        self.biasMatrices.clear()
        self.errorMatrices.clear()

        for i in range(1, self.length):
            weightMatrix = Matrix(self.structure[i], self.structure[i - 1])
            weightMatrix.randomise()
            self.weightMatrices.append(weightMatrix)

            biasMatrix = Matrix(self.structure[i], 1)
            biasMatrix.randomise()
            self.biasMatrices.append(biasMatrix)

        for i in range(self.length):
            errorMatrix = Matrix(self.structure[i], 1)  # All elements are zero by default
            self.errorMatrices.append(errorMatrix)

            layerMatrix = Matrix(self.structure[i], 1)  # All elements are zero by default
            self.layerMatrices.append(layerMatrix)

    def feed(self, _input, _rangeStart=0, _rangeEnd=None):
        """Applies the linear transformations to an initial vector and returns the output"""

        # Workaround since self.length cannot be accessed above.
        if _rangeEnd is None:
            _rangeEnd = self.length - 1

        assert _input.cols == 1 and _input.rows == self.structure[0]
        self.layerMatrices[0] = copy.copy(_input)
        for i in range(_rangeStart, _rangeEnd):
            self.layerMatrices[i + 1] = self.weightMatrices[i] * self.layerMatrices[i] + self.biasMatrices[i]
            self.layerMatrices[i + 1].setEach(lambda x, y, value: sigmoid(value))
        return self.layerMatrices[self.length - 1]

    def train(self, _input, _output, _rangeStart=0, _rangeEnd=None):
        """Trains the Network, given an input and expected output vector.
        Range is defined by _rangeStart and _rangeEnd (optional)"""

        # Workaround since self.length cannot be accessed above.
        if _rangeEnd is None:
            _rangeEnd = self.length - 1
        result = self.feed(_input)
        assert _output.rows == result.rows and _output.cols == result.cols == 1

        self.cost = 0
        self.globalError = 0
        for i in range(result.rows):
            gradient = sigmoidDerivative(result[i, 0])
            error = _output[i, 0] - result[i, 0]
            self.errorMatrices[_rangeEnd].data[i][0] = error * gradient
            self.cost += abs(error)

        if self.learningRate != 0:
            # Iterate though all layers
            for i in reversed(range(_rangeStart, _rangeEnd)):
                nextLayerErrors = self.errorMatrices[i + 1]

                # Iterate through the neurons in this layer
                for j in range(self.structure[i]):
                    localSum = 0

                    # Iterate through the neuron's connections to the next layer
                    for k in range(nextLayerErrors.rows):
                        self.weightMatrices[i][k, j] += self.learningRate * nextLayerErrors[k, 0] * \
                                                        self.layerMatrices[i][j, 0]
                        localSum += self.weightMatrices[i][k, j] * nextLayerErrors[k, 0]
                    currentError = localSum * sigmoidDerivative(self.layerMatrices[i][j, 0])
                    self.errorMatrices[i][j, 0] = currentError

                    if j < self.biasMatrices[i].rows:
                        self.biasMatrices[i][j, 0] += self.learningRate * currentError

                    self.globalError += abs(currentError)
        return result

    def serialise(self, _path):
        """Saves the network to a file"""

        print("Saving network to {}".format(_path), end="")
        with open(_path, "w") as f:
            f.write("STRUCTURE       #" + ",".join(str(x) for x in self.structure) + "\n")
            print(".", end="")      # Append a '.' for every stage as a visual indication of progress.
            f.write("WEIGHT MATRICES #" + "&".join(str(x) for x in self.weightMatrices) + "\n")
            print(".", end="")      # Append a '.' for every stage as a visual indication of progress.
            f.write("BIAS MATRICES   #" + "&".join(str(x) for x in self.biasMatrices) + "\n")
            print(".", end="")      # Append a '.' for every stage as a visual indication of progress.
            f.close()
        print(" Done.")

    def deserialise(self, _path):
        """Loads the network from a file"""

        print("Loading network from {}".format(_path), end="")
        self.weightMatrices.clear()
        self.biasMatrices.clear()
        self.errorMatrices.clear()
        self.structure.clear()

        with open(_path) as f:
            lines = f.read().split("\n")

            data = []
            for line in lines:
                if len(line) > 0:
                    data.append(line.split("#")[1])

            self.structure = [int(i) for i in data[0].split(",")]
            self.length = len(self.structure)
            self.initNetwork()
            print(".", end="")  # Append a '.' for every stage as a visual indication of progress.

            # Load weight matrices
            weightStrings = data[1][1:-1].split("&")      # '&' delimits different matrices. Removes the '[' and ']'.
            for i in range(len(weightStrings)):
                rows = weightStrings[i][1:-1].split(";")  # ';' delimits different rows. Removes the '[' and ']'.
                self.weightMatrices[i].setEach(
                    lambda y, x, val: float(rows[y].split(",")[x].replace("[", "").replace("]", "")))
                # Fill the matrix with the elements from the file
            print(".", end="")  # Append a '.' for every stage as a visual indication of progress.

            # Load bias matrices
            biasStrings = data[2][1:-1].split("&")         # '&' delimits different matrices. Removes the '[' and ']'.
            for i in range(len(biasStrings)):
                rows = biasStrings[i][1:-1].split(";")     # ';' delimits different rows. Removes the '[' and ']'.
                self.biasMatrices[i].setEach(
                    lambda y, x, val: float(rows[y].split(",")[x].replace("[", "").replace("]", "")))
                # Fill the matrix with the elements from the file
            print(".", end="")  # Append a '.' for every stage as a visual indication of progress.
        print(" Done.")
