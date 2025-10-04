#Batch normalization is performed in a mini batch wise fashion
#1) First, we compute the mean of each feature across the batch size (total number of samples)
#2) Compute the standard deviation for each features across the batch size
from msvcrt import putch

import torch
import numpy as np



def batch_normalization(data):
    means = []
    std_devs = []
    normalized_out = []
    for x in range(len(data)+1):
        out = data[:,x]
        out_mean = np.mean(out.numpy()) #mean across the array
        std_dev = np.std(out.numpy())
        means.append(out_mean)
        std_devs.append(std_dev)

        #normalize the data
        normalized_column = (out - out_mean) / std_dev
        output = data[:,x] = normalized_column #replace the original data column with the normalized column
        normalized_out.append(output)

    return normalized_out
    # mean_across_batch = [np.mean((data[:,x]).numpy()) for x in range(len(data)+1)]



#Layer normalization is performed across the elements of the entire row i.e. each sample

def layer_normalization(data):
    means = []
    std_devs = []
    normalized_out = []
    for x in range(data.shape[0]): #across the elements of the entire row
        out = data[:, x]
        out_mean = np.mean(out.numpy())  # mean across the array
        std_dev = np.std(out.numpy())
        means.append(out_mean)
        std_devs.append(std_dev)

        # normalize the data
        normalized_column = (out - out_mean) / std_dev
        output = data[:, x] = normalized_column  # replace the original data column with the normalized column
        normalized_out.append(output)

    return normalized_out





def main():
    data = torch.tensor([
        [1,2,3],
        [4,5,6]
    ]) #batch of 2 samples

    mean_mini_batch = batch_normalization(data)
    print(mean_mini_batch)

    layer_norm = layer_normalization(data)
    print(layer_norm)



if __name__ == '__main__':
    main()