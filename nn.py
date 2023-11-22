from numpy import genfromtxt
import numpy as np
import math

architecture = [
    {"input_dim": 5, "output_dim": 4, "activation": "relu"},
   # {"input_dim": 4, "output_dim": 6, "activation": "relu"},
   # {"input_dim": 6, "output_dim": 6, "activation": "relu"},
   # {"input_dim": 6, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 2, "activation": "sigmoid"},
]


def init_layers(architecture, seed = 99):
    np.random.seed(seed)
    number_of_layers = len(architecture)
    params_values = {}

    for index, layer in enumerate(architecture):
        current_index = index + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        
        params_values['w' + str(current_index)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(current_index)] = np.random.randn(
            layer_output_size, 1) * 0.1
        
    return params_values


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(dA, x):
    sig = sigmoid(x)
    return dA* sig * (1-sig)

def relu(x):
    return np.maximum(0,x)

def relu_prime(dA, x):
    if x.shape[0] != dA.shape[0]:
        x = np.transpose(x)
    dZ = np.array(dA, copy = True)
    dZ[x <= 0] = 0
    return dZ

"""
Outputs Error [a] is activation, [y] is target
""" 
def compute_error(A, Y):
    if A.shape[0] < 100:
        A = np.transpose(A)
    return np.mean((A - Y)**2)

def single_layer_step(a_prev,w_l,b_l):
    if a_prev.shape[0] > 100:
        a_prev = np.transpose(a_prev)
    a_prev = np.squeeze(np.asarray(a_prev))
    w_l = np.squeeze(np.asarray(w_l))
    
    z_l = np.dot(w_l, a_prev) + b_l 
    return z_l 


def forward_step(X, params_value, architecture):
    a_l = X

    for index, layer in enumerate(architecture):
        layer_index = index+1
        a_prev = a_l 
        w_l = params_value["w" + str(layer_index)]
        b_l = params_value["b" + str(layer_index)]
        z_l  = single_layer_step(a_prev, w_l, b_l)
        
        new_activation = relu(z_l) if layer["activation"]=="relu" else sigmoid(z_l)
        a_l = new_activation

        params_value["a" + str(layer_index)] = a_prev
        params_value["z" + str(layer_index)] = z_l

    return a_l, params_value


def single_layer_backward_step(dA, w_l,b_l,z_l,a_prev, activation):
    m = a_prev.shape[1]
   

    z_l = np.transpose(z_l)
    model_function = relu_prime(dA, z_l) if activation=="relu" else sigmoid_prime(dA, z_l)
    

    if a_prev.shape[1] != model_function.shape[0]:
        a_prev_transpose = np.transpose(a_prev)
        dW = np.dot(a_prev_transpose, model_function)/m
    else:    
        dW = np.dot(a_prev, model_function)/m

    model_function_transpose = np.transpose(model_function)
    db = np.sum(model_function_transpose, axis=1, keepdims=True)/m


    dA_prev = np.dot( model_function,w_l)
    
    dW = np.transpose(dW)
    return dW, db, dA_prev

def backward_step(architecture, a_l, target, params_value):
    step_values = {}

    squared_error = compute_error(a_l, target)
    a_l_transpose = np.transpose(a_l)
    dA_prev = 2*(np.subtract(a_l_transpose,target))

    for last_layer_index, layer in reversed(list(enumerate(architecture))):
        layer_index = last_layer_index + 1
        dA_update = dA_prev 

        a_prev = params_value["a" + str(layer_index)]
        z_l = params_value["z" + str(layer_index)]
        w_l = params_value["w" + str(layer_index)]
        b_l = params_value["b" + str(layer_index)]
        dW, db, dA_prev = single_layer_backward_step(dA_update, w_l, b_l, z_l, a_prev, layer["activation"])

        step_values["dW" + str(layer_index)] = dW
        step_values["db" + str(layer_index)] = db

    return step_values

def update_weights(params_values, step_values, architecture, learning_rate):
    for index, layer in enumerate(architecture):
        current_index = index+1
        params_values["w" + str(current_index)] -= learning_rate * step_values["dW"+ str(current_index)]        
        params_values["b"+ str(current_index)] -= learning_rate * step_values["db"+ str(current_index)]
    return params_values


def train(X,Y, architecture, epochs, learning_rate):
    params_values = init_layers(architecture)
    error_history = []

    for i in range(epochs):
        a_l, params_values = forward_step(X, params_values, architecture)
        transpose_a_l = np.transpose(a_l)
        error = compute_error(transpose_a_l,Y)
        error_history.append(error)

        step_values = backward_step(architecture,a_l,Y, params_values)
        params_values = update_weights(params_values,step_values,architecture,learning_rate)
    
    return params_values, error_history

def classify(Y_training):
    copy = np.zeros_like(Y_training)
    copy[np.arange(len(Y_training)), Y_training.argmax(1)] = 1
    return copy

def accuracy(training_output, test):
    count = 0
    length = len(training_output)
    for i in range(length):
        if np.array_equal(training_output[i],test[i]):
            count+=1
    return count/length


# train1 = genfromtxt('train1.csv', delimiter=',')
# train2 = genfromtxt('train2.csv', delimiter=',')
# test1 = genfromtxt('test1.csv', delimiter=',')
# test2 = genfromtxt('test2.csv', delimiter=',')


# #Dataset 1
# X_1 = train1[:, [0,1,2,3,4]]
# Y_1 = train1[:, [5,6]]
# X_test = test1[:, [0,1,2,3,4]]
# Y_test = test1[:, [5,6]]
# #print(Y_test)
# #Dataset 2
# X_2 = train2[:, [0,1,2,3,4]]
# Y_2 = train2[:, [5,6]]
# X_test_2 = test2[:, [0,1,2,3,4]]
# Y_test_2 = test2[:, [5,6]]

# #Train Data Set 1
# params_values, error_history = train(X_1,Y_1, architecture, 1000,0.00005)
# #Generate Results Data Set 1
# results, params_values = forward_step(X_test,params_values,architecture) 
# results = np.transpose(results)
# print(results)
# #Print Results
# results = classify(results)


# #Train Data Set 2
# params_values, error_history = train(X_2,Y_2, architecture, 1000,0.15)
# #Generate Results Data Set 2
# results, params_values = forward_step(X_test_2,params_values,architecture) 
# results = np.transpose(results)
# #Print Results
# #print(results)








