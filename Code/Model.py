import argparse

import torch
import torch.nn as nn
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
import numpy as np

from Blatt3 import FCLayer
from Blatt2 import BPE, Dictionary, FileHandler as FH2
from Blatt1 import FileHandler as FH1


def load_bpe(bpe_path: str):
    with open(f"../config/{bpe_path}", "rb") as f:
        return pkl.load(f)


def save_bpe(text_path: str, bpe_name: str, iterations: int):
    input = FH1.read_file_words(text_path)
    bpe = BPE(FH2.flattenList(input))
    bpe.learnBPE(iterations)
    with open(f"../config/{bpe_name}", "wb") as f:
        pkl.dump(bpe, f)
    return bpe


def load_voc(voc_path):
    with open(f"../config/{voc_path}", "rb") as f:
        return pkl.load(f)


def save_voc(text_path: str, voc_name: str, bpe):
    input = FH1.read_file_words(text_path)
    dict = Dictionary(bpe.applyBPE(input))
    with open(f"../config/{voc_name}", "wb") as f:
        pkl.dump(dict, f)
    return dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="path to text file")
    parser.add_argument("--bpe", type=str, help="path to bpe file")
    parser.add_argument("--voc", type=str, help="path to voc file")
    args = parser.parse_args()
    #bpe = saveBPE(args.text, args.bpe, 20)
    #print(bpe.getMerges())
    #bpe = load_bpe(args.bpe)
    #voc = saveVoc(args.text, args.voc, bpe)
    #voc = load_voc(args.voc)
    #print(voc.getDictionary())
    print(torch.flatten(torch.Tensor([[1,2],[3,4]]), start_dim=0))


if __name__ == "__main__":
    main()

class TranslationDataset(Dataset):
    def __init__(self, big_batch: list[tuple[np.ndarray, np.ndarray, np.ndarray]]):
        # big_batch has only a single element
        self.s_matr = big_batch[0][0]
        self.t_matr = big_batch[0][1]
        self.l_vec = big_batch[0][2]
    def __len__(self):
        # since each matrix has the same number of rows
        return len(self.s_matr)

    def __getitem__(self, index):
        return self.s_matr[index], self.t_matr[index], self.l_vec[index]

class FFTranslator(nn.Module):

    def __init__(self, dict_len, window_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(dict_len, embedding_dim)
        self.fc_source = FCLayer(embedding_dim*(2*window_size+1), 256, "relu")
        self.fc_target = FCLayer(embedding_dim*window_size, self.fc_source.shape_out, "relu")
        # concatenate the two outputs... with concat layer
        self.fc_1 = FCLayer(self.fc_source.shape_out + self.fc_target.shape_out, 384,  "relu")
        self.fc_2 = FCLayer(self.fc_1.shape_out, dict_len, "relu")

    def forward(self, item : tuple[np.array, np.array, np.array]):
        s = torch.from_numpy(item[0])
        t = torch.from_numpy(item[1])
        l = torch.from_numpy(item[2])
        # get embeddings
        s = self.embedding(s)
        t = self.embedding(t)
        # concatenate the embeddings
        torch.flatten(s, start_dim=0) # check for propagating with batches
        torch.flatten(t, start_dim=0)
        s = self.fc_source.propagate_forward(s)
        t = self.fc_target.propagate_forward(t)
        # concatenating s and t
        x = torch.cat((s, t), 0)
        x = self.fc_1.propagate_forward(x)
        x = self.fc_2.propagate_forward(x)
        # cross-entropy
        loss = nn.CrossEntropyLoss()
        return loss(x, l)
