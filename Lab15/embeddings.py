##Numerical representation for the text (caption)!

import torch 
from torch import nn
from torch.nn.utils.rnn import pad_sequence

##Define a vocabulary 
class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold #max no. of times a word must appear to be included in the vocab
        self.itos = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"} #itos -> index to string
        self.stoi = {v:k for k,v in self.itos.items()} #stoi -> reverse mapping of string to index
    
    def __len__(self):
        return len(self.itos)
    
    def tokenizer(self, text):
        return text.lower().split()
    
    def build_vocab(self, sentence_list):
        from collections import Counter
        freq = Counter()
        idx = 4 #the prev idx are occupied by special tokens 
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                freq[word] += 1
                if freq[word] == self.freq_threshold: #condition to be met to add in the vocab
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    def numericalize(self, text): #converts a single sentence into numbers
        tokenized = self.tokenizer(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized] #get the token, if word present in the vocab or use <unk>
    



#read the caption file and create a sentence list to pass to build_vocab method

caption_file = "/home/ibab/PycharmProjects/DL-Lab/Lab15/flickr8k/captions.txt"
captions = []
with open(caption_file,"r") as f:
    next(f) #skip the first line with header
    for line in f:
        img_name, caption = line.strip().split(",",1)
        captions.append(caption)


#call the vocab class
vocab = Vocabulary(freq_threshold=5)
vocab.build_vocab(captions)

# # numericalize all captions
# for caption in captions:
#     print(caption)
#     print(vocab.numericalize(caption))
numerical_captions = [vocab.numericalize(caption) for caption in captions]
print("Example numerical caption:", numerical_captions[0])
print("Vocabulary size:", len(vocab))
# print("stoi dictionary:", vocab.stoi)


# create an embedding using the pytorch embedding layer
vocab_size = len(vocab)
embed_size = 256
caption_embedder = nn.Embedding(vocab_size,embed_size)


#usage 
# numerical_captions = torch.LongTensor(numerical_captions)

#above approach throws error as the length of sequences differs for each list in the numerical captions list
#Apply padding to bring all the numerical lists to the same length
numerical_captions_tensors = [torch.LongTensor(cap) for cap in numerical_captions]
padded_caption = pad_sequence(numerical_captions_tensors, batch_first=True, padding_value=vocab.stoi["<PAD>"])

embedded_vectors = caption_embedder(padded_caption)
# print(embedded_vectors.shape)
torch.save(embedded_vectors, "captions_embeddings.pt")


