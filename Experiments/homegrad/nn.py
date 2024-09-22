import random
from homegrad.engine import *

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, activation = 'relu', nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin
        self.activation = activation
        if self.activation == 'none':
            self.nonlin = False

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        if self.nonlin:
            if self.activation == 'relu':
                return act.relu()
            if self.activation == 'tanh':
                return act.tanh()
        else:
            return act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        if self.nonlin:
            if self.activation == 'relu':
                return f'ReLu Neuron({len(self.w)})'
            elif self.activation == 'tanh':
                return f'Tanh Neuron({len(self.w)})'
        else:
            return f'Linear Neuron({len(self.w)})'


class Layer(Module):

    def __init__(self, nin, nout, activation='relu'):
        # Initialize neurons with the specified activation function
        self.neurons = [Neuron(nin, activation=activation) for _ in range(nout)]


    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLPerceptron(Module):

    def __init__(self, nin, nouts, activations=None):
        # If activations not specified, default to 'relu' for all except the last layer (output layer)
        if activations == None:
            self.activations = ['relu'] * (len(nouts) - 1) + ['none']
        else:
            self.activations = activations[0:len(nouts)-1] + ['none']

        # Create layers, each with its own specified activation function
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], activation=self.activations[i]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"