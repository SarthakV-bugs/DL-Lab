import numpy as np
import matplotlib.pyplot as plt

def weight_initilization(number_hidden_layers, num_neurons_each_layer, input_l):
    weights = []
    for i in range(number_hidden_layers):
        input_size = len(input_l)
        neurons = num_neurons_each_layer[i]
        prod = np.random.uniform(-1, 1, size=(input_size, neurons))
        input_l = [0] * neurons
        weights.append(prod)
    return weights

def ReLU(x):
    return np.maximum(0, x)

def softmax(x):
    exps = [np.exp(i) for i in x]
    sum_exps = sum(exps)
    return [j / sum_exps for j in exps]

def flow(number_hidden_layers, num_neurons_each_layer, input_l, weights, bias):
    x = np.array(input_l)
    all_outputs = []
    for i in range(number_hidden_layers):
        x = ReLU(np.dot(x, weights[i]) + bias[i])
        all_outputs.append(x.copy())
    return all_outputs, x  # Return all hidden outputs and final output

def main():
    number_hidden_layers = 2
    num_neurons_each_layer = [2, 1]

    # XOR-suitable weights and biases
    weights = [
        np.array([[1, 1], [1, 1]]),      # shape (2,2)
        np.array([[1], [-2]])            # shape (2,1)
    ]
    bias = [
        np.array([0, -1]),  # for layer 1
        np.array([0])       # for layer 2
    ]

    data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = [0, 1, 1, 0]

    # Collect hidden & output activations
    out_hidden_layers = [[] for _ in range(number_hidden_layers)]
    final_outputs = []

    for input_l in data:
        input_l = np.array(input_l)
        layer_outputs, final_out = flow(number_hidden_layers, num_neurons_each_layer, input_l, weights, bias)
        for layer_id, layer_output in enumerate(layer_outputs):
            out_hidden_layers[layer_id].append(layer_output)
        final_outputs.append(final_out)

    # Convert to arrays
    for j in range(number_hidden_layers):
        out_hidden_layers[j] = np.array(out_hidden_layers[j])
    final_outputs = np.array(final_outputs)

    # Plot hidden layer activations
    for k, layer_output in enumerate(out_hidden_layers):
        plt.figure()
        plt.title(f"Hidden Layer {k + 1} Outputs")
        for neuron_idx in range(layer_output.shape[1]):
            plt.plot(range(len(data)), layer_output[:, neuron_idx],
                     marker='o', label=f'Neuron {neuron_idx + 1}')
        plt.xlabel("Input Sample Index (XOR inputs)")
        plt.ylabel("Activation")
        plt.legend()
        plt.grid(True)
        plt.xticks(range(4), ['(0,0)', '(0,1)', '(1,0)', '(1,1)'])
        plt.show()

    # Plot final XOR output
    plt.figure()
    plt.title("Final XOR Output")
    plt.plot(range(len(data)), final_outputs.flatten(), marker='o', color='black', label='Output')
    plt.xlabel("Input Sample Index (XOR inputs)")
    plt.ylabel("Output Activation")
    plt.xticks(range(4), ['(0,0)', '(0,1)', '(1,0)', '(1,1)'])
    plt.grid(True)
    plt.legend()
    plt.show()

    # Decision boundary in input space
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    step = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = []

    for point in grid_points:
        _, out = flow(number_hidden_layers, num_neurons_each_layer, point, weights, bias)
        Z.append(out[0])  # final output is 1D

    Z = np.array(Z).reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['#FFAAAA', '#AAAAFF'], alpha=0.6)
    plt.colorbar(label='Output activation')

    # Original XOR points
    data = np.array(data)
    labels = np.array(labels)
    for i, point in enumerate(data):
        plt.scatter(point[0], point[1],
                    c='red' if labels[i] == 1 else 'black',
                    edgecolors='k', s=100,
                    label=f'{point}' if i < 2 else None)

    plt.title("Decision Boundary of XOR Network (Input Space)")
    plt.xlabel("Input x1")
    plt.ylabel("Input x2")
    plt.legend(["Class 1", "Class 0"])
    plt.grid(True)
    plt.show()

    # ✅ Plot decision regions in hidden layer 1 space
    hidden_layer_outputs = []
    final_preds = []

    for point in grid_points:
        hidden_outs, final_out = flow(number_hidden_layers, num_neurons_each_layer, point, weights, bias)
        hidden_layer_outputs.append(hidden_outs[0])  # first hidden layer
        final_preds.append(final_out[0])

    hidden_layer_outputs = np.array(hidden_layer_outputs)
    final_preds = np.array(final_preds)

    plt.figure(figsize=(6, 5))
    plt.title("Decision Boundary in Hidden Layer 1 Space")

    plt.scatter(hidden_layer_outputs[:, 0], hidden_layer_outputs[:, 1],
                c=final_preds, cmap='bwr', alpha=0.6, edgecolors='k', s=20)

    plt.xlabel("Neuron 1 Activation")
    plt.ylabel("Neuron 2 Activation")
    plt.grid(True)

    # XOR points projected into hidden layer 1 space
    for i, point in enumerate(data):
        hidden_out, _ = flow(number_hidden_layers, num_neurons_each_layer, point, weights, bias)
        plt.scatter(hidden_out[0][0], hidden_out[0][1],
                    color='yellow' if labels[i] == 1 else 'black',
                    edgecolors='k', s=150, label=f'{point}')

    plt.legend()
    plt.show()

    print("Final layer output:\n", final_outputs)

if __name__ == '__main__':
    main()
