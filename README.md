# Airplane Classifier

For the AE1205 Python programming competition, we made an airplane classifier. Using (deep) neural networks, that we
programmed ourselves, the aim was to output the manufacturer, family, or variant of an aircraft, given its picture.

All code and data files can also be found on [Github](https://github.com/MikeKuijper/Aircraft-Classifier).

NOTE:
Due to the size limitation on the project submission, I have included only the absolute essentials for running one of the
two applications - excluding a third time-consuming one. Please get the full codebase from GitHub, as that is what the majority
of the work went to.


## Approach in general

To start, we programmed a basic gradient descent algorithm, together with forward propagation for a given network structure.
A network with 20 input nodes, then two hidden layers with respectively 10 and 8 nodes and 4 nodes in the output layer, 
would, for example, be represented as ``[20, 10, 8, 4]``. Note that the number of elements in the input vectors is fixed.

### Feed forward
The network applies a linear transformation for every layer transition. Thus, we can see every layer as a vector in
n-dimensional space, where the n denotes the number of nodes in the layer. The elements of the standard matrix for the 
linear transformation are often referred to as weights. After the linear transformation, the vector is translated with a
bias vector. This is to assure that the network can have an output, when encountering a zero vector. 

After the translation,
the entire vector is 'normalised' using, in this case, the sigmoid function. This is to assure that the values stay within
reasonable bounds and increases the likelihood of convergence. It also means that outputting exactly 0 and 1 is theoretically
impossible, since the sigmoid function has them as horizontal asymptotes. Practically speaking, however,
with the precision of floats (or doubles outside of python), it suffices to aim for it anyway.

This process is done for every layer. Thus, the entire network might be seen as a vector function, that returns a 
transformed vector. The key to using Neural Networks, is to give the values of the vectors a macroscopic meaning. For instance,
you might interpret the value of an element of the output vector as the probability that the input vector fits a certain classification.

### Gradient descent / Backpropagation
In order to determine the weights and biases for the network, it must train on data. The method for that consists of two parts,
the feed forward part, and then the 'backpropagation' step. Feed forward was discussed above; backpropagation is a method
adjusting the weights and biases, so that the network more accurately predicts an output.

Backpropagation works from the output layer towards the input layer - hence the name. It requires an input vector and a
'correct' output vector. The algorithm then compares the calculated output (found using the feed forward) to the correct output.
For each element, one could then ask: "How could I improve this answer by adjusting weights and biases?" If the answer needs
to increase, then one of the weights must likely increase as well. The same argument could be made for biases. Using this, it
is possible to make a new 'correct' answer for the previous layer, and repeat the process for the entire network.

### Train phase
It is important to realise that this method only optimises a single input and output pair. Training a network for a single 
input vector is computationally easy, but also useless. The aim, of course, is to make a network that accurately predicts
vectors that it has never seen before. For this, the network requires lots and lots of datapoints. By repeatedly training
the network on those datapoints, eventually the entire network 'fits' to the data. There's a sweet spot there, since it is
also possible to overfit your network, but that is beyond the scope of this explanation.

### Learning Rate
During the backpropagation step, every adjustment is scaled by the learning factor. Most training issues can be traced 
back to a wrong setting for this value. Too high might prevent the network from reaching a minimum, let along a global one. 
Too low and the training might take ages, besides getting stuck in local minima. Typically, it's best to gradually decrease 
it as training progresses. This is done by multiplying it by another factor, the learningRateDecay factor, which will lead 
to an exponential decay.

## Specifics
This specific project relies on the [FGVC Dataset by S. Maji et al.](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/#aircraft),
consisting of 100 images for 100 different airplane variants along with data on their
manufacturer (i.e. Airbus, Boeing), family (i.e. Airbus A320, Boeing 747), variant (i.e. Airbus A319, Boeing 747-800), and
bounding box data for cropping.

These images were cropped, edge detected using the horizontal Sobel operator, converted to grayscale, and resized to 64x32
pixels (in that order). After this, they were converted to input vectors by placing the normalised pixel value in a 2048-element
vector (horizontal then vertical).

The output vectors depend slightly on the exact configuration. In any case, the output vectors are n-dimensional, where 
n denotes the number of different classifications. The correct vector is the one with all zeroes, except for the element
corresponding to the model. For instance, if classifying between Airbus, Boeing, and Cessna, all Airbuses would have the correct
output of ``[1, 0, 0]``, whereas the Boeing and Cessna images would have ``[0, 1, 0]`` and ``[0, 0, 1]`` respectively. Thus,
the output value can be interpreted as the probability or confidence that the given image is of the type corresponding to the element.


### File structure
The root folder contains three python files. ``MachineLearning.py`` is the library containing the logic described above.
``train.py`` was used to generate and train the networks. ``gui.py`` is the pygame implementation, where you can drag and
drop your own image file into and input into the network, which will output the classification.

The data folder contains networks that were trained with different configurations in the form of ``.dat`` files. For instance, one could attempt to differentiate
between Airbus and Boeing airplanes, whereas another might attempt to output the families for those.

### Configurations
The naming convention for the network files is as follows: ``'network_{}-{}-{}.dat'.format(a, b, c)`` where a is the number
of pixels width, b the subset file minus the extension (i.e. airbusboeing), and c the detail level (either m, f or v for
manufacturer, family and variant respectively). Included are the following trained configurations:
1. **Airbus/Boeing-F** was trained to output the families of all included Airbus and Boeing airplanes. Assessed score: 35.9% (7.1% expected if chosen randomly)
    ```python
       classPath = "data/airbusboeing.txt"
       networkPath = "data/network_64-airbusboeing-f.dat"
       detailLevel = 2

2. **Airbus/Boeing-M** was trained to differentiate between Airbus and Boeing airplanes. Assessed score: 70.4% (50% expected if chosen randomly)
    ```python
       classPath = "data/airbusboeing_m.txt"
       networkPath = "data/network_64-airbusboeing_m-m.dat"
       detailLevel = 1
3. **All-M** was trained to output the manufacturer of all airplanes included in the dataset. Assessed score: 31.3% (3.3% expected if chosen randomly)
   ```python
       classPath = "data/manufacturers.txt"
       networkPath = "data/network_64-manufacturers-m.dat"
       detailLevel = 1
4. **All-F** was trained to output the families of all airplanes included in the dataset. Assessed score: 14.2% (1.4% expected if chosen randomly)
    ```python
       classPath = "data/families.txt"
       networkPath = "data/network_64-families-f.dat"
       detailLevel = 2
   
## Usage
There are a few different ways provided to play around with the networks. 
1. First, ``gui.py`` visually displays the result of any odd image you drag into the window.
2. Secondly, ``train.py`` allows you to train and assess your own neural networks on a classification problem.
3. Lastly, ``data/formatImages.py`` allows you to entirely start from scratch with a custom dataset, or a different resolution for the current one.

Be patient when you start ``gui.py``. It has to load the neural network before it can display anything.

For the first two, you will have to choose a configuration above, or start from scratch (only for ``train.py``). Simply paste the code
above in spot designated with ``# Configuration``. For the third, you will have to download the entire [FGVC Dataset by S. Maji et al.](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/#aircraft);
Due to the impracticality of sending over gigabytes worth of photos, they are not included.

## Reflection
The performance of the networks leave a lot to be desired. In practice, they don't perform as well as in the isolated 
testing environment. That is likely due to the following reasons:
* Python performs relatively slow. Therefore, training takes long to progress. All configurations were trained over the course of **multiple days**.
* Images in practice aren't cropped the same way as they are in the training data, since there is no bounding box.
* A neural network used like this is not ideal. Typically, one would use a convolutional neural network for this, but the backpropagation of those is quite tricky.
* Time. Given another few weeks, the networks would have performed significantly better.
* The image dataset was slightly biased. Since every variant had 100 pictures, manufacturers with more variants were overrepresented in the dataset. This especially affects _Airbus/Boeing-M_, since it caused quite a nasty Boeing bias.
* Difficult task. It would have performed better if all images were from roughly the same angle. This way, it is essentially encoding a rough 3D model, which is a difficult task.
* Few images. Ideally, we would have had many more images. However, datasets like FGVC are not common. And manually compiling thousands of images would take much too long.
* Old. The dataset does not contain newer aircraft, such as the Boeing 787.

## Requirements
As stipulated by the competition requirements, the code does not require any exotic libraries. However, it does require
Pillow, which we asked Prof. Hoekstra about, who said it should be okay. To install it, execute ``$ pip install Pillow`` in your console or, even better, ``$ pip install -r requirements.txt`` from the root folder.