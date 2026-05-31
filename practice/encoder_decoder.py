import pandas as pd
from collections import Counter
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import spacy


#configurations
Max_length = 20


#define a custom dataset loader
class Machine_translation_dataset(Dataset):
    def __init__(self, csv_file, src_col="source", tgt_col="target", max_len=Max_length):
        self.data = pd.read_csv(csv_file)
        self.src_col = src_col
        self.tgt_col = tgt_col
        self.max_len = max_len

        #clean the dataset and normalize the text
        self.data[self.src_col] = self.data[self.src_col].astype(str).str.lower().str.strip()
        self.data[self.tgt_col] = self.data[self.tgt_col].astype(str).str.lower().str.strip()

        #Tokenizers
        self.english = spacy.load("en_core_web_sm")
        self.hindi = spacy.load("hi")

        #build vocabularies
        self.src2idx, self.idx2src = self.build_vocab(self.data[self.src_col], self.english)
        self.tgt2idx, self.idx2tgt = self.build_vocab(self.data[self.tgt_col], self.hindi)


    def tokenize(self, tokenizer, sentence):
        return [token.text.lower() for token in tokenizer(sentence)]

    def build_vocab(self, texts, tokenizer, min_freq=1):
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(tokenizer, text)
            counter.update(tokens)

        vocab =



def main():
    csv_file = "Hindi_English_Truncated_Corpus.csv"

if __name__ == '__main__':
        main()