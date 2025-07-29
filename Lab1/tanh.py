import numpy as np
import matplotlib.pyplot as plt


def tan_h(z):
    return (np.exp(z) - np.exp(-z)) /  (np.exp(z) + np.exp(-z))

def tan_h_derivative(z):
    return 1 - (tan_h(z))**2

def main():
    z = np.linspace(-10,10,100)
    print(tan_h(z))
    y = tan_h(z)
    print(tan_h_derivative(z))
    f = tan_h_derivative(z)


    print(f"mean of the activation function output:",np.mean(y))

    plt.figure(figsize=(8, 6))
    plt.plot(z, y , label="tanh Function")
    plt.plot(z, f, label="Derivative of tanh function")
    plt.title("tanh Function and it's derivative")
    plt.xlabel("z")
    plt.ylabel("value")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()