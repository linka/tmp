"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
import numpy as np
import os


class Activation(object):

    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result, i.e. the output of forward (it will be tested).
    """

    # No additional work is needed for this class, as it acts like an abstract base class for the others

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):

    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):

    """
    Sigmoid non-linearity
    """

    # Remember do not change the function signatures as those are needed to stay the same for AL

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        # Might we need to store something before returning?
        result = np.divide(1, (1 + np.exp(-x)))
        self.state = result
        return result

    def derivative(self):
        # Maybe something we need later in here...
        return np.multiply(self.state, 1-self.state)


class Tanh(Activation):

    """
    Tanh non-linearity
    """

    # This one's all you!

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        e_x = np.exp(x)
        e_x2 = np.exp(-x)
        result = np.divide(np.subtract(e_x, e_x2),  np.add(e_x, e_x2))
        self.state = result
        return result

    def derivative(self):
        return 1 - np.power(self.state, 2)


class ReLU(Activation):

    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        result = np.maximum(x, 0.)
        assert x.shape == result.shape
        self.state = result
        return result

    def derivative(self):
        return np.where(self.state > 0, 1., 0.)

# Ok now things get decidedly more interesting. The following Criterion class
# will be used again as the basis for a number of loss functions (which are in the
# form of classes so that they can be exchanged easily (it's how PyTorch and other
# ML libraries do it))


class Criterion(object):

    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):

    """
    Softmax loss
    """

    def __init__(self):

        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None

    def forward(self, x, y):

        self.logits = x
        self.labels = y

        # ...
        a = np.max(x, axis=0, keepdims=True)
        log_sum = a + np.log(np.sum(np.exp(np.subtract(x, a)), axis=1, keepdims=True))
        
        loss = -1 * np.sum(np.multiply(y, np.subtract(x, log_sum)), axis=1)
        self.sm = np.exp(np.subtract(x, log_sum))

        return loss

    def derivative(self):
        # self.sm might be useful here...
        return self.sm - self.labels


class BatchNorm(object):

    def __init__(self, fan_in, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        # inference parameters
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):

        if eval:
            self.norm = np.subtract(self.x, self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma * self.norm + self.beta

        self.x = x

        self.mean = 1./self.x.shape[0] * (np.sum(x, axis=1, keepdims=True))
        self.var = 1./self.x.shape[0] * np.sum(np.power(x - self.mean, 2), axis=1, keepdims=True)
        self.norm = np.subtract(self.x, self.mean) / np.sqrt(self.var + self.eps)
        self.out = self.gamma * self.norm + self.beta

        # update running batch statistics
        self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * self.mean
        self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.var

        return self.out

    def backward(self, delta):
        dxhat = np.multiply(delta, self.gamma)
        self.dbeta = np.sum(delta, axis=0, keepdims=True)
        self.dgamma = np.sum(np.multiply(delta, dxhat), axis=0, keepdims=True)
        dvar = -0.5 * np.sum(np.multiply(np.power(self.var + self.eps, -1.5),
                                         np.multiply(dxhat, np.subtract(self.x, self.mean))), axis=0, keepdims=True)
        dmean = (-1 * np.sum(np.multiply(dxhat, np.power(self.var + self.eps, -0.5)), axis=0, keepdims=True) \
                 - 2/m * dvar * np.sum(np.subtract(self.x, self.mean), axis=0, keepdims=True))

        m = self.x.shape[0]
        dx = (dxhat * np.power(self.var + self.eps, -0.5)
              + dvar * (2/m * np.subtract(self.x, self.mean))
              + dmean * 1/m)
        return dx

# These are both easy one-liners, don't over-think them
def random_normal_weight_init(d0, d1):
    return np.random.normal(size=(d0, d1))


def zeros_bias_init(d):
    return np.zeros(d)


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn, bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly
        self.W = [weight_init_fn(d0, d1) for (d0, d1) in zip([input_size] + hiddens, hiddens + [output_size])]
        self.dW = [np.zeros((d0, d1)) for (d0, d1) in zip([input_size] + hiddens, hiddens + [output_size])]
        self.b = [bias_init_fn(d0) for d0 in hiddens + [output_size]]
        self.db = [np.zeros(d0) for d0 in hiddens + [output_size]]
        # HINT: self.foo = [ bar(???) for ?? in ? ]

        # if batch norm, add batch norm parameters
        if self.bn:
            self.bn_layers = None

        # Feel free to add any other attributes useful to your implementation (input, output, ...)
        self.x = None
        self.m = 0
        self.errors = []
        self.losses = []
        self.out = None

    def forward(self, x):
        self.errors = []
        self.losses = []
        self.x = x
        self.m = x.shape[0]

        tmp = x
        for i in range(self.nlayers):
            tmp = np.dot(tmp, self.W[i])
            tmp = np.add(tmp, self.b[i])
            tmp = self.activations[i].forward(tmp)
        self.out = tmp
        return tmp

    def zero_grads(self):
        self.dW = [np.zeros(grad.shape) for grad in self.dW]
        self.db = [np.zeros(grad.shape) for grad in self.db]

    def step(self):
        self.W = [w - self.lr / self.m * dw for (w, dw) in zip(self.W, self.dW)]
        self.b = [b - self.lr / self.m * db for (b, db) in zip(self.b, self.db)]

    def backward(self, labels):
        # output
        loss = self.criterion.forward(self.out, labels)
        err = np.mean(np.argmax(self.out, axis=1) != labels)
        self.losses.append(loss)
        self.errors.append(err)
        dL = self.criterion.derivative()

        dPrev = dL
        for i in range(self.nlayers-1, 0, -1):
            dZ = self.activations[i].derivative()
            self.dW[i] = 1. / self.m * np.dot(dZ, dPrev.T)
            self.db[i] = 1. / self.m * np.sum(dZ, axis=1, keepdims=True)
            dPrev = np.dot(self.W[i].T, dZ)
        return dPrev

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, test = dset
    trainx, trainy = train
    valx, valy = val
    testx, testy = test

    idxs = np.arange(len(trainx))

    training_losses = []
    training_errors = []
    validation_losses = []
    validation_errors = []

    # Setup ...


    for e in range(nepochs):

        # Per epoch setup ...
        for b in range(0, len(trainx), batch_size):
            # Train ...
            mlp.forward(trainx[b:b+batch_size])
            mlp.backward(trainy[b:b+batch_size])

            mlp.step()

            training_losses = mlp.losses
            training_errors = mlp.errors

        for b in range(0, len(valx), batch_size):
            mlp.eval()
            mlp.forward(valx[b:b+batch_size])
            # Val ...
            validation_losses = mlp.losses
            validation_errors = mlp.errors

        # Accumulate data...
    print(e)
    # Cleanup ...
    mlp.zero_grads()

    for b in range(0, len(testx), batch_size):
        mlp.eval()
        mlp.forward(valx[b:b + batch_size])
        # Test ...

    # Return results ...

    return (training_losses, training_errors, validation_losses, validation_errors)
