import argparse

import torch
import torch.nn as nn
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
import numpy as np

from Blatt3 import FCLayer
from Blatt2 import BPE, Dictionary, Batch, FileHandler as FH2
from Blatt1 import FileHandler as FH1

debug = 0


def load_bpe(bpe_path: str):
    with open(f"../config/{bpe_path}", "rb") as f:
        return pkl.load(f)


def save_bpe(text_path_de: str, text_path_en: str, bpe_name: str, iterations: int):
    input_de = FH1.read_file_words(text_path_de)
    input_en = FH1.read_file_words(text_path_en)
    bpe = BPE(FH2.flattenList(input_de + input_en))
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


def load_batch(batch_path):
    with open(f"../config/{batch_path}", "rb") as f:
        return pkl.load(f)


def save_voc(text_path_de: str, text_path_en: str, voc_name: str, bpe):
    input_de = FH1.read_file_words(text_path_de)
    input_en = FH1.read_file_words(text_path_en)
    dict = Dictionary(bpe.applyBPE(input_de + input_en))
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
    max_patience = 3  # maybe change later
    patience = max_patience

    for epoch in range(epochs):
        # set model to training mode to ensure that the gradient tracking is on
        model.train()
        print(f"Epoch {epoch + 1}")
        train_one_epoch(model, train_dataloader, optimizer, checkpoints[epoch], lossfunc, epoch)

        # evaluate the model every n epochs and check for stagnation
        if (epoch + 1) % eval_every_n_epochs == 0:
            eval_perpl, eval_acc = eval_model(model, eval_dataloader, lossfunc, len(eval_dataloader))
            print(f"Perplexity: {eval_perpl} ,Accuracy {eval_acc}")
            # first evaluation
            if best_perpl == -1 or best_acc == -1:
                best_perpl = eval_perpl
                best_acc = eval_acc
            # check if the curr perpl and accuracy is smaller
            if eval_perpl < best_perpl or eval_acc > best_acc:  # maybe change to and later
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
                    print(
                        f"Stopping training, model not improving. Best perplexity: {best_perpl}, Best accuracy: {best_acc}, Epoch: {epoch + 1}")
                    break


def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    optimizer: torch.optim,
                    checkpoints: list[int],
                    lossfunc: nn.Module, epoch: int):
    eval_every_n_batches = 10
    # loss calculation here
    for i, batch in enumerate(dataloader):
        model.train()
        _, _, y = batch
        y = y.squeeze()  #loss func needs a row vector,
        y = y.to(torch.long)  #coverted to long
        y_w = model(batch)
        if debug > 1:
            print(y_w.shape, y.shape)
        loss = lossfunc(y_w, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if debug > 2:
            if i % eval_every_n_batches == 0:
                # dont eval full dataset every time
                eval_perpl, eval_acc = eval_model(model, dataloader, lossfunc, 50)
                print(f"Batch {i}/{len(dataloader)}: Perplexity: {eval_perpl}, Accuracy: {eval_acc}")
        if i in checkpoints:
            save_model(model, f"model_epoch{epoch}_iter_{i}")


def eval_model(model: nn.Module,
               data_loader: DataLoader,
               loss: nn.Module,
               number_of_batches: int):
    model.eval()
    # disable gradient calculation
    total_loss = 0
    total_len = 0
    total_acc = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i == number_of_batches:
                break
            _, _, y = batch
            y_w = model(batch)
            y = y.squeeze()
            y = y.to(torch.long)
            total_loss += loss(y_w,
                               y).item()  # total_loss isnt a vector since the function calculates the mean of the losses for a batch
            total_len += 1
            # torch.max is (values, indices)
            _, predicted = torch.max(y_w, dim=1)  # takes the index of the highest value of each row
            total_acc += (predicted == y).sum().item()
    # return perplexity and accuracy
    return 2 ** (total_loss / total_len), (total_acc / (total_len * data_loader.batch_size))


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
        self.fc_1 = FCLayer(self.fc_source.shape_out + self.fc_target.shape_out, 512, "relu")
        #self.fc_2 = FCLayer(self.fc_1.shape_out, 256, "relu")
        self.fc_3 = FCLayer(self.fc_1.shape_out, dict_len, "relu")  # Adding an extra layer

    def forward(self, item: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        s, t, _ = item
        # get embeddings
        s = self.embedding(
            s)  # embedding size stays the same, input is now a matrix of values, every value gets replaced by a vector
        t = self.embedding(t)  # => 3 dim
        if debug > 1:
            print(f"embedding shape: {s.shape}")
        # concatenate the embeddings
        s = torch.flatten(s, start_dim=1)  # concatenating the embedding vectors of a row (which is the inner dimension)
        t = torch.flatten(t, start_dim=1)
        """
        we could also transpose if our fc layer would multiply W*x^t and not like pytorch x*W^t
        but (W*x^t)^t = x*W^t so its equivalent 
        """
        if debug > 1:
            print(f"flattened shape: {s.shape} (after transpose)")
        s = self.fc_source.propagate_forward(s)
        t = self.fc_target.propagate_forward(t)
        # concatenating s and t
        # since s and t have shapes (batch_size, n) concatenating them along dim=1
        # results in a shape of (batch_size, ...)
        if debug > 1:
            print(f"fc shape: {s.shape} {t.shape}")
        x = torch.cat((s, t), 1)
        if debug > 1:
            print(f"concatenated shape: {x.shape}")
        x = self.fc_1.propagate_forward(x)
        #x = self.fc_2.propagate_forward(x)
        x = self.fc_3.propagate_forward(x)
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
    parser.add_argument("--train_batch", type=str, help="name of training batch file")
    parser.add_argument("--eval_batch", type=str, help="name of evaluation batch file")
    args = parser.parse_args()

    # load or create BPE
    if args.bpe is not None:
        bpe = load_bpe(args.bpe)
    else:
        bpe = save_bpe(args.train_de, args.train_en, "bpe.pkl", 7000)

    # load or create vocabulary
    if args.voc is not None:
        voc = load_voc(args.voc)
    else:
        voc = save_voc(args.train_de, args.train_en, "voc.pkl", bpe)

    # load or create Batches
    if args.train_batch is not None and args.eval_batch is not None:
        train_batch = load_batch(args.train_batch)
        eval_batch = load_batch(args.eval_batch)
    else:
        # load training and evaluation data
        raw_train_data_de = FH1.read_file_words(args.train_de)
        raw_train_data_en = FH1.read_file_words(args.train_en)
        raw_eval_data_de = FH1.read_file_words(args.eval_de)
        raw_eval_data_en = FH1.read_file_words(args.eval_en)
        if debug > 0:
            print("applying bpe...")
        # apply BPE to data and apply vocabulary
        train_data_de = voc.filterInput(bpe.applyBPE(raw_train_data_de))
        train_data_en = voc.filterInput(bpe.applyBPE(raw_train_data_en))
        eval_data_de = voc.filterInput(bpe.applyBPE(raw_eval_data_de))
        eval_data_en = voc.filterInput(bpe.applyBPE(raw_eval_data_en))
        if debug > 0:
            print("creating batches...")
        # create big batches
        train_total_words = sum([len(x) for x in train_data_en]) + len(train_data_en)
        eval_total_words = sum([len(x) for x in eval_data_en]) + len(eval_data_en)
        windowsize = 5
        train_batch = Batch(train_data_de, train_data_en, train_total_words, windowsize, voc, voc)
        eval_batch = Batch(eval_data_de, eval_data_en, eval_total_words, windowsize, voc, voc)

        # saving batches for next time
        with open(f"../config/train_batch.pkl", "wb") as f:
            pkl.dump(train_batch, f)
        with open(f"../config/eval_batch.pkl", "wb") as f:
            pkl.dump(eval_batch, f)

    if debug > 4:
        str_train_batch = Batch.getBatchesWithStrings(train_batch.batches, voc, voc)
        FH2.saveBatches(str_train_batch, "train_batch.txt", 20)
        print(str_train_batch)

    # create datasets
    train_dataset = TranslationDataset(train_batch.batches)
    eval_dataset = TranslationDataset(eval_batch.batches)

    if True:
        # test with smaller dataset
        indices = torch.arange(0, 10000)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        eval_dataset = train_dataset
        print(f"debug: train dataset length: {len(train_dataset)}")

    # create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=128, shuffle=False)
    if debug > 0:
        print("start training...")
    # training
    model = FFTranslator(train_batch.voc_len, train_batch.windowsize, 100)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    epochs = 100
    checkpoints = [[] for i in range(epochs)]
    training_routine(model, train_dataloader, eval_dataloader, optimizer, epochs, True, checkpoints, 1)
