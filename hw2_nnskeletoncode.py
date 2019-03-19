## this is example skeleton code for a Tensorflow/PyTorch neural network 
## module. You are not required to, and indeed probably should not
## follow these specifications exactly. Just try to get a sense for the kind
## of program structure that might make this convenient to implement.

# overall module class which is subclassed by layer instances
# contains a forward method for computing the value for a given
# input and a backwards method for computing gradients using
# backpropogation.
import numpy as np

class Module():
    def __init__(self):
        self.prev = None # previous network (linked list of layers)
        self.output = None # output of forward call for backprop.

    learning_rate = 1E-2 # class-level learning rate

    def __call__(self, input):
        if isinstance(input, Module):
            # chain two networks together with module1(module2(x))
            # update prev and output
            self.prev = input
            self.output = self.forward(self.prev.output)
        else:
            # evaluate on an input.
            # update output
            self.output = self.forward(input)

        return self

    def forward(self, *input):
        raise NotImplementedError

    def backwards(self, *input):
        raise NotImplementedError


# sigmoid non-linearity
class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, input):
        # compute sigmoid, update fields
        return 1.0 / (1.0 + np.exp(-input))

    def backwards(self, gradient):
        # compute gradients with backpropogation and data from forward pass
        return gradient * (self.output * (1 - self.output))



# linear (i.e. linear transformation) layer
class Linear(Module):
    def __init__(self, input_size, output_size, is_input=False):
        super(Linear, self).__init__()
        # todo. initialize weights and biases.
        self.W = np.random.rand(input_size, output_size) * 0.01
        self.bias = np.zeros((output_size))

    def forward(self, input):  # input has shape (batch_size, input_size)
        # todo compute forward pass through linear input
        self.prev = input
        self.output = np.dot(input, self.W) + np.expand_dims(self.bias, axis=0)
        return self.output

    def backwards(self, gradient):
        # todo compute and store gradients using backpropogation
        #update
        self.W = self.W - np.dot(np.transpose(self.input, (1, 0)), gradient) * learning_rate
        self.bias = self.bias - gradient  * learning_rate
        #
        return np.dot(gradient, np.transpose(self.W, (1, 0)))


# generic loss layer for loss functions
class Loss:
    def __init__(self):
        self.prev = None

    def __call__(self, input):
        self.prev = input
        return self

    def forward(self, input, labels):
        raise NotImplementedError

    def backwards(self):
        raise NotImplementedError


# MSE loss function
class MeanErrorLoss(Loss):
    def __init__(self):
        super(MeanErrorLoss, self).__init__()

    def forward(self, input, labels):  # input has shape (batch_size, input_size)
        # compute loss, update fields
        return np.sum(((input - labels) ** 2.0), axis=1)

    def backwards(self):
        # compute gradient using backpropogation
        return (input - labels) * 2.0


## overall neural network class
class Network(Module):
    def __init__(self):
        super(Network, self).__init__()
        # todo initializes layers, i.e. sigmoid, linear
        self.fc1 = Linear(2, 2)
        self.sig1 = Sigmoid()
        self.fc2 = Linear(2, 2)


    def forward(self, input):
        # todo compute forward pass through all initialized layers
        return self.fc2(self.sig1(self.fc1(input)))
        
        
    def backwards(self, grad):
        # todo iterate through layers and compute and store gradients
        x = self.fc2.backwards(grad)
        x = self.sig1.backwards(grad)
        x = self.fc1.backwards(grad)


    def predict(self, data):
        # todo compute forward pass and output predictions
        _ = self.fc2(self.sig1(self.fc1(data)))
        return self.fc2.output


    def accuracy(self, test_data, test_labels):
        # todo evaluate accuracy of model on a test dataset
        pred = self.predict(test_data)
        assert test_labels.shape[1] == 1 and pred.shape[1] == 1 and test_labels.shape[0] == pred.shape[0]
        return (pred == test_labels).sum() / test_labels.shape[0]


# function for training the network for a given number of iterations
def train(model, data, labels, num_iterations, minibatch_size, learning_rate):
    # todo repeatedly do forward and backwards calls, update weights, do 
    # stochastic gradient descent on mini-batches.
