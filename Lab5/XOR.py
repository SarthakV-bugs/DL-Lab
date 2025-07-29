import numpy as np
import matplotlib.pyplot as plt

def weight_initilization(number_hidden_layers, num_neurons_each_layer, input_l):
    weights = []  # 3D matrix to store the weights as a dot prod. of input layer and the neurons
    for i in range(number_hidden_layers):  # iterated each layer
        input_size = len(input_l)
        neurons = num_neurons_each_layer[i]  # number of neurons in each layer
        # print(f"here: {neurons}")
        prod = np.random.uniform(-1, 1, size=(input_size, neurons))
        input_l = [0] * neurons  # updates the input with each layer, therefore put a check condition here
        weights.append(prod)

    return weights  # Total 20 weights and two 3D layered matrix


def ReLU(x):
    return np.maximum(0, x)


def softmax(x):
    exps = [np.exp(i) for i in x]
    sum_exps = sum(exps)
    return [j / sum_exps for j in exps]





def flow(number_hidden_layers, num_neurons_each_layer, input_l, weights,bias):

     #to store the outputs of each layer
    x = np.array(input_l)
    all_outputs = []
    for i in range(number_hidden_layers):
        x = ReLU(np.dot(x, weights[i])+bias[i]) #output for each layer
        all_outputs.append(x.copy())
        print(f"Layer {i+1} results:", x)
    return all_outputs,x

# def flow(number_hidden_layers, num_neurons_each_layer, input_l, weights,bias):
#
#      #to store the outputs of each layer
#     x = np.array(input_l)
#     for i in range(number_hidden_layers):
#         x = ReLU(np.dot(x, weights[i])+bias[i]) #output for each layer
#         print(f"Layer {i+1} results:", x)
#     return x


###doesnt update the current layer for the input
# def flow(number_hidden_layers, num_neurons_each_layer, input_l, weights,bias):
#
#     x = np.array(input_l)
#     for i in range(number_hidden_layers):
#         a = ReLU(np.dot(x, weights[i])+bias[i])
#
#         print(f"Layer {i+1} results:", a)
#     return a
    # print(softmax(x))


def main():
    ##initiaL input layer
    number_hidden_layers = 2  ##int
    num_neurons_each_layer = [2, 1]  # list of numbers denoting neurons in each hidden layer
    weights = [
        np.array([[1, 1], [1, 1]]),  # shape (2,2)
        np.array([[1], [-2]])  # shape (2,1)
    ]

    bias = [
        np.array([0,-1]),
        np.array([0])
    ]

    # outside = []
    # data = [[0,0],[0,1],[1,0],[1,1]]
    # for i in range(len(data)):
    #     input_l = data[i]
    #     input_l = np.array(input_l)
    #     # print(input_l)
    #     # weights = [[[1,1],[1,1]],[1,-2]]
    #
    #     # print(flow(number_hidden_layers, num_neurons_each_layer, input_l, weights))
    #     y = flow(number_hidden_layers, num_neurons_each_layer, input_l, weights,bias)
    #     outside.append(y)
    # print(f"Final layer output", np.array(outside))  outside = []


    out_hidden_layers = [[] for _ in range(number_hidden_layers)] #output for each layer is stored here

    data = [[0,0],[0,1],[1,0],[1,1]]
    for i in range(len(data)):
        input_l = data[i]
        input_l = np.array(input_l)
        # print(input_l)
        # weights = [[[1,1],[1,1]],[1,-2]]

        # print(flow(number_hidden_layers, num_neurons_each_layer, input_l, weights))
        layers_outs, final_out = flow(number_hidden_layers, num_neurons_each_layer, input_l, weights,bias)

        for layer_id, layer_outputs in enumerate(layers_outs):
            out_hidden_layers[layer_id].append(layer_outputs)

        for j in range(number_hidden_layers):
            out_hidden_layers[j] = np.array( out_hidden_layers[j])

        for k, layer_output in enumerate(out_hidden_layers):
            plt.figure()
            plt.title(f"Hidden Layer {i + 1} Outputs")
            for neuron_idx in range(layer_output.shape[1]):
                plt.plot(range(len(data)), layer_output[:, neuron_idx], marker='o', label=f'Neuron {neuron_idx + 1}')
            plt.xlabel("Input Sample Index")
            plt.ylabel("Activation")
            plt.legend()
            plt.grid(True)
            plt.show()




    #     outside.append(y)
    # print(f"Final layer output", np.array(outside))











    # ##vector DS
    # number_hidden_layers = 2  ##int
    # num_neurons_each_layer = [2, 1]  # list of numbers denoting neurons in each hidden layer

    # print(weight_initilization(number_hidden_layers, num_neurons_each_layer, input_l))
    # weights = weight_initilization(number_hidden_layers, num_neurons_each_layer, input_l)
    # print(type(weights))




if __name__ == '__main__':
    main()