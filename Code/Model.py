import argparse

import torch
import torch.nn as nn
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
import numpy as np

from Blatt3 import FCLayer
from Blatt2 import BPE, Dictionary, Batch, FileHandler as FH2
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


def save_model(model: nn.Module, model_name: str):
    torch.save(model, f"../config/{model_name}")


def load_model(model_name: str):
    return torch.load(f"../config/{model_name}")


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
    print(torch.flatten(torch.Tensor([[1, 2], [3, 4]]), start_dim=0))


def training_routine(model: nn.Module,
                     train_dataloader: DataLoader,
                     eval_dataloader: DataLoader,
                     optimizer: torch.optim,
                     epochs: int,
                     halve_lr: bool,
                     checkpoints: list[list[int]],
                     eval_every_n_epochs: int):
    lossfunc = nn.CrossEntropyLoss()
    best_perpl = -1
    best_acc = -1
    max_patience = 5 # maybe change later
    patience = max_patience

    for epoch in range(epochs):
        # set model to training mode to ensure that the gradient tracking is on
        model.train()
        print(f"Epoch {epoch + 1}")
        train_one_epoch(model, train_dataloader, optimizer, checkpoints[epoch], lossfunc)

        # evaluate the model every n epochs and check for stagnation
        if (epoch + 1) % eval_every_n_epochs == 0:
            eval_perpl,eval_acc  = eval_model(model, eval_dataloader, lossfunc)
            #new_eval should be smaller than old eval
            # first evaluation
            if best_perpl == -1 or best_acc == -1:
                best_perpl = eval_perpl
                best_acc = eval_acc
            # check if the curr perpl and accuracy is smaller
            if eval_perpl < best_perpl or eval_acc > best_acc: # maybe change to and later
                best_perpl = min(best_perpl, eval_perpl)
                best_acc = max(best_acc, eval_acc)
                patience = max_patience
            else:
                patience -= 1
            if patience <= 0:
                if halve_lr:
                    # optimizer.param_groups is a list of the different weight groups, since we only have one group..
                    optimizer.param_groups[0]["lr"] /= 2
                    print(f"Reducing learning rate to {optimizer.param_groups[0]['lr']}")
                    patience = max_patience
                else:
                    # stop training if model isnt improving and we shouldnt halve the lr
                    print(f"Stopping training, model not improving. Best perplexity: {best_perpl}, Best accuracy: {best_acc}, Epoch: {epoch + 1}")
                    break






def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    optimizer: torch.optim,
                    checkpoints: list[int],
                    lossfunc: nn.Module):
    # loss calculation here
    for batch in enumerate(dataloader):
        loss = torch.zeros(len(batch))
        for i in range(len(batch)):
            _,_,y_m = batch[i]
            y = model.forward(batch[i])
            loss[i] = lossfunc(y,y_m)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()




def eval_model(model: nn.Module,
               data_loader: DataLoader,
               loss: nn.Module):
    model.eval()
    # disable gradient calculation
    total_loss = 0
    total_len = 0
    total_acc = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            for item in data:
                _,_,y = item
                y_m = model.forward(item)
                total_loss += loss(y_m,y) # maybe later += loss.item()
                total_len += 1
                # torch.max is (values, indices)
                _,x = torch.max(y_m, 0) # maybe change to dim=1 later
                if x == y:
                    total_acc += 1
    # return perplexity and accuracy
    return 2**(total_loss/total_len), (total_acc/total_len)





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
        self.fc_source = FCLayer(embedding_dim * (2 * window_size + 1), 256, "relu")
        self.fc_target = FCLayer(embedding_dim * window_size, self.fc_source.shape_out, "relu")
        # concatenate the two outputs... with concat layer
        self.fc_1 = FCLayer(self.fc_source.shape_out + self.fc_target.shape_out, 384, "relu")
        self.fc_2 = FCLayer(self.fc_1.shape_out, dict_len, "relu")

    def forward(self, item: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        s, t, l = item
        # get embeddings
        s = self.embedding(s) # embedding size stays the same, input is now a matrix of values, every value gets replaced by a vector
        t = self.embedding(t) # => 3 dim
        # concatenate the embeddings
        torch.flatten(s, start_dim=0)  # check for propagating with batches
        torch.flatten(t, start_dim=0)
        s = self.fc_source.propagate_forward(s)
        t = self.fc_target.propagate_forward(t)
        # concatenating s and t
        # since s and t have shapes (batch_size, n) concatenating them along dim=1
        # results in a shape of (batch_size, ...)
        x = torch.cat((s, t), 1)
        x = self.fc_1.propagate_forward(x)
        x = self.fc_2.propagate_forward(x)
        # cross-entropy
        #loss = nn.CrossEntropyLoss()
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_de", type=str, help="path to de training data")
    parser.add_argument("--train_en", type=str, help="path to en training data")
    parser.add_argument("--eval_de", type=str, help="path to de evaluation data")
    parser.add_argument("--eval_en", type=str, help="path to en evaluation data")
    parser.add_argument("--bpe", type=str, help="name of bpe file")
    parser.add_argument("--voc", type=str, help="name of voc file")
    args = parser.parse_args()

    # load or create BPE
    if args.bpe is not None:
        bpe = load_bpe(args.bpe)
    else:
        bpe = save_bpe(args.train, "bpe.pkl", 7000)

    # load or create vocabulary
    if args.voc is not None:
        voc = load_voc(args.voc)
    else:
        voc = save_voc(args.train, "voc.pkl", bpe)

    # load training and evaluation data
    raw_train_data_de = FH1.read_file_words(args.train_de)
    raw_train_data_en = FH1.read_file_words(args.train_en)
    raw_eval_data_de = FH1.read_file_words(args.eval)
    raw_eval_data_en = FH1.read_file_words(args.eval)

    # apply BPE to data and apply vocabulary
    train_data_de = voc.filterInput(bpe.applyBPE(raw_train_data_de))
    train_data_en = voc.filterInput(bpe.applyBPE(raw_train_data_en))
    eval_data_de = voc.filterInput(bpe.applyBPE(raw_eval_data_de))
    eval_data_en = voc.filterInput(bpe.applyBPE(raw_eval_data_en))

    # create big batches
    train_total_words = sum([len(x) for x in train_data_en])
    eval_total_words = sum([len(x) for x in eval_data_en])
    train_batch = Batch.createBatches(train_data_de, train_data_en, train_total_words, 5, voc, voc)
    eval_batch = Batch.createBatches(eval_data_de, eval_data_en, eval_total_words, 5, voc, voc)

    # create datasets
    train_dataset = TranslationDataset(train_batch)
    eval_dataset = TranslationDataset(eval_batch)

    # create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size= 128, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size= 128, shuffle=False)
