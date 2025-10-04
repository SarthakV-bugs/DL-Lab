import numpy as np

def weight_initialization(number_hidden_layers, num_neurons_each_layer, input_l):
    
    weights = [] #3D matrix to store the weights as a dot prod. of input layer and the neurons
    for i in range(number_hidden_layers): #iterated each layer 
        input_size = len(input_l)
        neurons = num_neurons_each_layer[i] #number of neurons in each layer
        # print(f"here: {neurons}")
        prod = np.random.uniform(-1,1,size=(input_size,neurons))
        input_l = [0] * neurons #updates the input with each layer, therefore put a check condition here
        weights.append(prod)

    return weights #Total 20 weights and two 3D layered matrix


def ReLU(x):

    return np.maximum(0,x) 



def softmax(x):
    exps = [np.exp(i) for i in x]
    sum_exps = sum(exps)
    return [j / sum_exps for j in exps]



res = []
def flow(number_hidden_layers, num_neurons_each_layer, input_l, weights):
    x = np.array(input_l)
    for i in range(number_hidden_layers):
        x = ReLU(np.dot(x, weights[i]))
        print(f"Layer 1 results:", res.append(x))
    print(softmax(x))

        



def main():

   ##initiaL input layer
    input_l =  [-2.4,1.2,-0.8, 1.1]
    input_l = np.array(input_l) ##vector DS
    number_hidden_layers = 2 ##int
    num_neurons_each_layer = [3,2] #list of numbers denoting neurons in each hidden layer

    print(weight_initialization(number_hidden_layers, num_neurons_each_layer, input_l))
    weights = weight_initialization(number_hidden_layers, num_neurons_each_layer, input_l)
    print(type(weights))

    print(flow(number_hidden_layers, num_neurons_each_layer,input_l,weights))

if __name__ == '__main__':
    main()