import numpy as np
import matplotlib.pyplot as plt


def softmax(z):
    l = []
    for x in z:
        l.append(np.exp(x) / np.sum(np.exp(z)))
    return np.array(l)


def softmax_derivative(l):
    ##creates a matrix of output from softmax function
    jacob =  [] #matrix
    for i in range(len(l)): #loops over the elements in the output list
        t = []
        for j in range(len(l)): #for each row
            if i == j: #diagonal element
                t.append((l[j]*(1-l[j])))
            else: #non diagonal element
                t.append((-l[i]*l[j]))
        jacob.append(t)
    return np.matrix(jacob)



def main():
    # z = np.linspace(-10,10,100)
    z = [0,0]
    l = softmax(z)
    print(l)
    # l = [0.65,0.24,0.098]
    print(softmax_derivative(l))
    x = softmax_derivative(l)



    plt.figure(figsize=(8, 6))
    plt.plot(z, l , label="Softmax Function")
    # plt.plot(z, x )
    plt.title("softmax Function and it's derivative")
    plt.xlabel("z")
    plt.ylabel("value")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()