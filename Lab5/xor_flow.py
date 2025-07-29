import numpy as np


##structure
##create the structure first using loops
##initialize the weights first, use random numbers to allot and create a 3d matrix

##flow of the network




def weights(input_l, number_hidden_layers,num_neurons_each_layer):
    """

    :param input_l: dynamic list,updates as the output of each layer
    :param number_hidden_layers: int denoting the obvious
    :param num_neurons_each_layer: list with indexing representing each layer
    :return: 3D matrix of weights for the entire network
    """


    weight_matrix = [] #list to hold the 2D matrix generated as a result of each layer
    for i in range(number_hidden_layers):
        neurons = num_neurons_each_layer[i] #number of neurons in ith layer
        input_layer = len(input_l)
        w = [] #list to hold the outputs of each layer
        for j in range(neurons):
            output = []
            for k in range(input_layer):
                output.append(np.random.uniform(-1,1))
            w.append(np.array(output))
        input_l = [0] * neurons    #updating the input layer as per the number of neurons of the current layer as it will be the same size as the output
        weight_matrix.append(np.array(w))
    return weight_matrix

def ReLU(x):
    return np.maximum(0,x)

def softmax(x):
    exps = [np.exp(i) for i in x]
    sum_exps = sum(exps)
    return [j / sum_exps for j in exps]


def flow(input_l, number_hidden_layers, num_neurons_each_layer, weight_matrix):

    ##initialize an output holder for each layer if needed
    # out = []
    for i in range(number_hidden_layers):
        ##access the first layer for computation
        wt = weight_matrix[i]
        # print(l1_wt.shape)
        # print(l1_wt)

        inp = input_l  # first layer of input, we need to update after each layer
        n_neurons = num_neurons_each_layer[i]
        # loop each neuron of the current layer
        out = []
        for j in range(n_neurons):
            z = 0  # sum initialization
            # iterate each input of the input layer
            for k in range(len(inp)):
                z += wt[j][k] * inp[k]  # for one neuron the entire summation
            a = ReLU(z)  # takes a singular value as an input
            out.append(a)
        input_l = out
    return softmax(out)



def main():
    ##initiaL input layer
    number_hidden_layers = 2  ##int
    num_neurons_each_layer = [2, 1]  # list of numbers denoting neurons in each hidden layer
    weights = [[[1, 1], [1, 1]], [1, -2]]
    data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for i in range(len(data)):
        input_l = data[i]
        input_l = np.array(input_l)
        # print(input_l)
        # weights = [[[1,1],[1,1]],[1,-2]]

        print(flow(number_hidden_layers, num_neurons_each_layer, input_l, weights))


if __name__ == '__main__':
    main()