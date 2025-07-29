import numpy as np
import matplotlib.pyplot as plt


def leaky_ReLU(z):
    return np.maximum(0.01*z,z)

def leaky_relu_derivative(z):
    return np.where(z>0,1,0.01)

def main():
    z = np.linspace(-10,10,100)
    g = leaky_ReLU(z)
    h = leaky_relu_derivative(z)
    print(leaky_ReLU(z))
    print(leaky_relu_derivative(z))

    plt.figure(figsize=(8, 6))
    plt.plot(z, g , label="leaky ReLU Function")
    plt.plot(z, h, label="Derivative of leaky ReLU function")
    plt.title("ReLU Function and it's derivative")
    plt.xlabel("z")
    plt.ylabel("value")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()