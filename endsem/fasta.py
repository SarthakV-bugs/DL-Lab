from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, random_split


#load sample.fasta file using custom dataset
def fasta_reader(file_path):

    seq_label = []
    sequences = []
    with open(file_path, "r") as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip()

            if line.startswith(">"):
                line = line.replace(">", "")
                seq_label.append(line)
            else:
                sequences.append(line)

    return seq_label, sequences


#define a custom dataset for reading fasta files

class fasta(Dataset):
    def __init__(self, file_path):
        seq_label, sequences = fasta_reader(file_path)
        self.seq_label = seq_label
        self.sequences = sequences

    def __len__(self):
        return len(self.seq_label)

    def __getitem__(self, idx):
        return self.seq_label[idx], self.sequences[idx]



def main():
    file_path = "/home/ibab/Desktop/DL_datasets/sample.fasta"
    print(fasta_reader(file_path))

    dataset = fasta(file_path)

    #split the dataset into train and test
    n = len(dataset)
    train_len = int(n * 0.8)
    test_len = n - train_len
    train_set, test_set = random_split(dataset, [train_len, test_len])
    print(f"training dataset : {len(train_set)} \ntesting dataset : {len(test_set)}")

    #define the dataloader
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=2, shuffle=False)

    print(len(train_loader), len(test_loader))

    #samples in all batches
    for batch, data in enumerate(train_loader):
        print(batch)
        print(data)
        break


if __name__ == "__main__":
    main()






##when mutliple sequence lines are present
# def fasta_reader(file_path):
#     labels = []
#     sequences = []
#     current_seq = []
#
#     with open(file_path, "r") as f:
#         for line in f:
#             line = line.strip()
#
#             if line.startswith(">"):
#                 # If we already collected a sequence, save it
#                 if current_seq:
#                     sequences.append("".join(current_seq))
#                     current_seq = []
#
#                 labels.append(line)  # store header line (label)
#
#             else:
#                 current_seq.append(line)
#
#         # Save the last sequence after finishing the loop
#         if current_seq:
#             sequences.append("".join(current_seq))
#
#     return labels, sequences
#
# print(fasta_reader(file_path))