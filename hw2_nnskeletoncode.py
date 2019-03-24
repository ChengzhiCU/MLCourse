## this is example skeleton code for a Tensorflow/PyTorch neural network 
## module. You are not required to, and indeed probably should not
## follow these specifications exactly. Just try to get a sense for the kind
## of program structure that might make this convenient to implement.

# overall module class which is subclassed by layer instances
# contains a forward method for computing the value for a given
# input and a backwards method for computing gradients using
# backpropogation.
import numpy as np
import scipy.io as sio

class Module():
    def __init__(self):
        self.prev = None # previous network (linked list of layers)
        self.output = None # output of forward call for backprop.

    learning_rate = 1E-3 # class-level learning rate

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


class Optimizer():
    def __init__(self):
        pass

class Adam(Optimizer):
    def __init__(self, param):
        super(Adam, self).__init__()
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.m = np.zeros_like(param)
        self.v = np.zeros_like(param)
        self.eps = 1e-9
    
    def update(self, grad):
        # print(grad)
        self.m, self.v = self.beta1 * self.m + (1 - self.beta1) * grad, self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        return self.m / (np.sqrt(self.v) + self.eps)

# linear (i.e. linear transformation) layer
class Linear(Module):
    def __init__(self, input_size, output_size, is_input=False):
        super(Linear, self).__init__()
        # todo. initialize weights and biases.
        self.W = np.random.rand(input_size, output_size) * 0.1
        self.bias = np.zeros((output_size))
        self.input = None
        self.gradW = None
        self.gradb = None

    def forward(self, input):  # input has shape (batch_size, input_size)
        # todo compute forward pass through linear input
        self.input = input
        return np.dot(input, self.W) + np.expand_dims(self.bias, axis=0)

    def backwards(self, gradient):
        # todo compute and store gradients using backpropogation
        #update
        self.gradW = np.dot(np.transpose(self.input, (1, 0)), gradient)
        self.gradb = np.mean(gradient, axis=0)

        # SGD
        # self.W = self.W - gradW * learning_rate
        # self.bias = self.bias - gradb * learning_rate
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
        self.input = input.output
        self.labels = labels
        return np.mean(np.sum(((input.output - labels) ** 2.0), axis=1))

    def backwards(self):
        # compute gradient using backpropogation
        return (self.input - self.labels) * 2.0


## overall neural network class
class Network(Module):
    def __init__(self):
        super(Network, self).__init__()
        # todo initializes layers, i.e. sigmoid, linear
        self.fc1 = Linear(2, 128)
        self.sig1 = Sigmoid()
        self.fc2 = Linear(128, 512)
        self.sig2 = Sigmoid()
        self.fc3 = Linear(512, 3)
        self.sig3 = Sigmoid()

        self.val_tobe_optimd = [self.fc1.W, self.fc1.bias, self.fc2.W, self.fc2.bias, self.fc3.W, self.fc3.bias]
        self.val_grads = [self.fc1.gradW, self.fc1.gradb, self.fc2.gradW, self.fc2.gradb, self.fc3.gradW, self.fc3.gradb]
        self.optims = [Adam(x) for x in self.val_tobe_optimd]


    def forward(self, input):
        # todo compute forward pass through all initialized layers
        return self.sig3(self.fc3(self.sig2(self.fc2(self.sig1(self.fc1(input))))))
        
        
    def backwards(self, grad):
        # todo iterate through layers and compute and store gradients
        x = grad
        x = self.sig3.backwards(x)
        x = self.fc3.backwards(x)
        x = self.sig2.backwards(x)
        x = self.fc2.backwards(x)
        x = self.sig1.backwards(x)
        x = self.fc1.backwards(x)

        self.val_grads = [self.fc1.gradW, self.fc1.gradb, self.fc2.gradW, self.fc2.gradb, self.fc3.gradW, self.fc3.gradb] 


    def update(self):
        for val, grad, optim in zip(self.val_tobe_optimd, self.val_grads, self.optims):
            # print('grad here', val.shape, grad.shape)
            val -= Module.learning_rate * optim.update(grad)

    def predict(self, data):
        # todo compute forward pass and output predictions
        _ = self.fc3(self.sig2(self.fc2(self.sig1(self.fc1(data)))))
        return self.fc3.output


    def accuracy(self, test_data, test_labels):
        # todo evaluate accuracy of model on a test dataset
        pred = self.predict(test_data)
        assert test_labels.shape[1] == 1 and pred.shape[1] == 1 and test_labels.shape[0] == pred.shape[0]
        return (pred == test_labels).sum() / test_labels.shape[0]


class Dataloader():
    def __init__(self, dataset_ind):

        combo = sio.loadmat("hw2_data.mat")
        if dataset_ind == 1:
            self.x = combo['X1']
            self.y = combo['Y1'] / 255.
        elif dataset_ind == 2:
            self.x = combo['X2']
            self.y = combo['Y2'] / 255.
        elif dataset_ind == 3:     # Debug quadratic func
            n = 1000
            self.x = np.array([[i / n, i / n] for i in range(n)])
            self.y = np.array([[i * i / n / n] for i in range(n)])
        else:
            raise ValueError("No such dataset {}".format(dataset_ind))
        self.len = self.x.shape[0]

        self.cur_order = np.random.permutation(self.len)
        self.data_start = 0

    def next_batch(self, batch_size):
        x, y = None, None
        if self.data_start + batch_size >= self.len:
            batch_size = self.len - self.data_start - 1
            x = self.x[self.cur_order[self.data_start:self.data_start+batch_size]]
            y = self.y[self.cur_order[self.data_start:self.data_start+batch_size]]
            self.data_start = 0
        else:
            x = self.x[self.cur_order[self.data_start:self.data_start + batch_size]]
            y = self.y[self.cur_order[self.data_start:self.data_start + batch_size]]
            self.data_start += batch_size
        return x, y

# function for training the network for a given number of iterations
def train(model, dataloader, loss, num_iterations, minibatch_size, learning_rate):
    # todo repeatedly do forward and backwards calls, update weights, do 
    # stochastic gradient descent on mini-batches.
    
    for iter in range(num_iterations):
        x, y = dataloader.next_batch(minibatch_size)
        # x = np.zeros_like(x)    # Debug
        pred = model.forward(x)

        # print('prediction', pred.output)
        loss_value = loss.forward(pred, y)
        # print("After forward")
        
        loss_grad = loss.backwards()
        # print("After calculate loss")

        model.backwards(loss_grad)
        # print("After backprop")

        model.update()
        # print("After update weights")

        print(loss_value)
        # print("loss: {:.5}".format(loss_value))

if __name__ == '__main__':

    dataloader = Dataloader(2)
    model = Network()
    loss = MeanErrorLoss()
    # loss_v = loss()
    # loss=None

    train(model, dataloader, loss, 100000, 128, 1e-5)





