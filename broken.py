from numpy import genfromtxt
import numpy as np
import math

architecture = [
        {"input_dim": 4, "output_dim": 32, "activation": "relu"},
    # {"input_dim": 4, "output_dim": 6, "activation": "relu"},
    # {"input_dim": 6, "output_dim": 6, "activation": "relu"},
    # {"input_dim": 6, "output_dim": 4, "activation": "relu"},
        {"input_dim": 32, "output_dim": 2, "activation": "sigmoid"},
    ]


def sigmoid( x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(dA, x):
    sig = sigmoid(x)
    return dA* sig * (1-sig)

def relu(x):
    return np.maximum(0,x)

def relu_prime(dA, x):
    #if x.shape[0] != dA.shape[0]:
    #    x = np.transpose(x)
    dZ = np.array(dA, copy = True)
    dZ[x <= 0] = 0
    return dZ

class NN():
    def __init__(self, architecture= [
        {"input_dim": 4, "output_dim": 3, "activation": "relu"},
        {"input_dim": 3, "output_dim": 2, "activation": "sigmoid"},
    ]):
        
        self.architecture = architecture

    def init_layers(self, seed = 99):
        np.random.seed(seed)
        self.number_of_layers = len(self.architecture)+1
        self.params_values = {}

        for index, layer in enumerate(self.architecture):
            current_index = index + 1
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]
            
            self.params_values['w' + str(current_index)] = np.random.randn(
                layer_output_size, layer_input_size) * 0.1
            self.params_values['b' + str(current_index)] = np.random.randn(
                layer_output_size, 1) * 0.1


    def single_layer_step(self, a_prev,w_l,b_l):
        if a_prev.shape[0] > 100:
            a_prev = np.transpose(a_prev)
        a_prev = np.squeeze(np.asarray(a_prev))
        w_l = np.squeeze(np.asarray(w_l))
        
        z_l = np.dot(w_l, a_prev) + b_l 
        return z_l 


    def forward_step(self, X):
        a_l = X
        for index, layer in enumerate(architecture):
            print(a_l)
            layer_index = index+1
            print(layer_index)
            a_prev = a_l 
            w_l = self.params_values["w" + str(layer_index)]
            b_l = self.params_values["b" + str(layer_index)]
            z_l  = self.single_layer_step(a_prev, w_l, b_l)
            
            new_activation = relu(z_l) if layer["activation"]=="relu" else sigmoid(z_l)
            a_l = new_activation

            self.params_values["a" + str(layer_index)] = a_prev
            self.params_values["z" + str(layer_index)] = z_l
            #print(a_l)

        return a_l












