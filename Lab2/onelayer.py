import numpy as np


def input_layer(x):
    """

    :param x:
    :return: input as a vector
    """
    return np.array(x)

def unit(x,w):
    """

    :param x: the input data
    :param w: the weights initialized
    :return: returns value after applying activation function
    """
    w = np.array(w)
    z = np.transpose(x).dot(w)
    return max(0,z)

def main():
    x = [0.2,-1.3,2,0.01]
    w = [0.001,0.01,-0.005,-1.2]

    print(unit(x,w))

if __name__ == '__main__':
    main()