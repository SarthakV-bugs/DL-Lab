import numpy as np


def string_to_vector(input):
    """
    :param input: string
    :return: list of one hot encoded vector for each char
    """

    unique_chars = sorted(set(input.upper()))
    # vocab_size = len(unique_chars)
    # print(vocab_size)
    char_to_index = {char:i for i, char in enumerate(input.upper())}
    print(char_to_index)

    one_hot_list = []
    for char in input:
        # print(char)
        one_hot_vector = np.zeros((len(input), 1))
        print(one_hot_vector.shape)
        if char.upper() in char_to_index:

            one_hot_vector[char_to_index[char.upper()]] = 1
        one_hot_list.append(one_hot_vector)
    return np.array(one_hot_list)


def vanillaRNN(x_t, h_t_pre, W_x_h, W_h_h, W_y_h):
    """
    :param x_t: takes the input at a time step
    :param h_t_pre: previous hidden state vector
    :param W_x_h: weight matrix for input and hidden state at time t
    :param W_h_h: weight matrix for hidden state at t-1 to t
    :param W_y_h: weight matrix for output at hidden state t
    :return: hidden state at t and output at that time step
    """

    # h_t = tan_h(np.sum((np.dot(W_h_h,h_0),np.dot(W_x_h,x_t))))
    c1 = np.dot(W_h_h,h_t_pre)
    c2 = np.dot(W_x_h,x_t)
    h_t = c1 + c2
    h_t = np.tanh(h_t)
    # print(h_t)

    y_t = np.dot(W_y_h,h_t)
    # print(y)
    return h_t, y_t


def main():

    h_0 = np.array([0,0,0]) #initial hidden state vector
    W_x_h = np.array([[0.5,-0.3],
             [0.8,0.2],
             [0.1,0.4]]) #3*2

    W_h_h = np.array([[0.1,0.4,0.0],
             [-0.2,0.3,0.2],
             [0.05,-0.1,0.2]])

    W_h_y = np.array([[1.0,-1.0,0.5],
             [0.5,0.5,-0.5]])


    input = 'hello'
    X = string_to_vector(input)



    # X = np.array([[1,2],[-1,1]]) #input vector with encoded input, need to generalize it

    hidden_states = [h_0]
    outputs = []


    h_t = h_0
    # update the hidden state from h_0 to h_t
    for t in range(len(X)):
        print(f"At time_stamp {t + 1}, input x_t : {X[t]}")
        h_t,y_t = vanillaRNN(X[t],h_t,W_x_h,W_h_h,W_h_y)
        hidden_states.append(h_t)
        outputs.append(y_t)
        print(f"Hidden states:{hidden_states}")
        print(f"Outputs{outputs}")

    return hidden_states, outputs


if __name__ == '__main__':
    main()