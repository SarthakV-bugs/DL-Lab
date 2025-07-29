import numpy as np
import matplotlib.pyplot as plt


def ReLU(z):
    return np.maximum(0,z)
    # m = []

    #conditional way
    # for x in z:
    #     if x > 0:
    #         m.append(x)
    #     else:
    #         m.append(0)
    # return m

def relu_derivative(z):
    return np.where(z>0,1,0)

def main():
    z = np.linspace(-10,10,100)
    g = ReLU(z)
    h = relu_derivative(z)
    print(ReLU(z))


    print(f"mean of the activation function output:",np.mean(g))

    print(relu_derivative(z))

    plt.figure(figsize=(8, 6))
    plt.plot(z, g , label="ReLU Function")
    plt.plot(z, h, label="Derivative of ReLU function")
    plt.title("ReLU Function and it's derivative")
    plt.xlabel("z")
    plt.ylabel("value")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()