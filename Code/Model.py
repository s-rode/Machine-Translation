import argparse

import torch
import torch.nn as nn
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
import numpy as np
import yaml

from Blatt3 import FCLayer
from Blatt2 import BPE, Dictionary, Batch, FileHandler as fh2
from Blatt1 import FileHandler as fh1

debug = 0

# hyperparameters
EMBEDDING_DIM = None
EPOCHS = None
CHECKPOINTS_WBATCH = None
CHECKPOINTS_WEPOCH = None
LEARNING_RATE = None
EVAL_EVERY_N_BATCHES = None
OWN_LINEAR = None
WINDOW_SIZE = None
FC_OUT_ST = None
FC_OUT_1 = None
BATCH_SIZE_TRAIN = None
BATCH_SIZE_EVAL = None
BPE_MERGES = None
PATIENCE = None
DROPOUT = None
HALVE_LR = None



def load_bpe(bpe_path: str):
    with open(f"../config/{bpe_path}", "rb") as f:
        return pkl.load(f)


def save_bpe(text_path_de: str, text_path_en: str, bpe_name: str, iterations: int):
    input_de = fh1.read_file_words(text_path_de)
    input_en = fh1.read_file_words(text_path_en)
    bpe = BPE(fh2.flattenList(input_de + input_en))
    bpe.learnBPE(iterations)
    with open(f"../config/{bpe_name}", "wb") as f:
        pkl.dump(bpe, f)
    return bpe


def save_model(model: nn.Module, model_name: str):
    print(f"saving model to {model_name}")
    torch.save(model, f"../config/models/{model_name}")


def load_model(model_name: str):
    return torch.load(f"../config/models/{model_name}")


def load_voc(voc_path):
    with open(f"../config/{voc_path}", "rb") as f:
        return pkl.load(f)


def load_batch(batch_path):
    with open(f"../config/{batch_path}", "rb") as f:
        return pkl.load(f)


def save_batch(bpe: BPE, voc: Dictionary, args_de, args_en, window_size, batch_name: str):
    raw_data_de = fh1.read_file_words(args_de)
    raw_data_en = fh1.read_file_words(args_en)
    if debug > 0:
        print("applying bpe...")
    # apply BPE to data and apply vocabulary
    data_de = voc.filterInput(bpe.applyBPE(raw_data_de))
    data_en = voc.filterInput(bpe.applyBPE(raw_data_en))
    if debug > 0:
        print("creating batches...")
    # create big batches
    total_words = sum([len(x) for x in data_en]) + len(data_en)
    batch = Batch(data_de, data_en, total_words, window_size, voc, voc)
    # saving batches for next time
    with open(f"../config/{batch_name}", "wb") as f:
        pkl.dump(batch, f)
    return batch


def save_voc(text_path_de: str, text_path_en: str, voc_name: str, bpe):
    input_de = fh1.read_file_words(text_path_de)
    input_en = fh1.read_file_words(text_path_en)
    dict = Dictionary(bpe.applyBPE(input_de + input_en))
    with open(f"../config/{voc_name}", "wb") as f:
        pkl.dump(dict, f)
    return dict


def training_routine(model: nn.Module,
                     train_dataloader: DataLoader,
                     eval_dataloader: DataLoader,
                     optimizer: torch.optim,
                     epochs: int,
                     halve_lr: bool,
                     checkpoints: list[list[int]],
                     eval_every_n_batches: int):
    loss_func = nn.CrossEntropyLoss()
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}, Shape: {param.shape}")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        train_one_epoch(model, train_dataloader, eval_dataloader, optimizer, checkpoints[epoch+1], loss_func, epoch, eval_every_n_batches, halve_lr)


def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    eval_dataloader: DataLoader,
                    optimizer: torch.optim,
                    checkpoints: list[int],
                    lossfunc: nn.Module, epoch: int, eval_every_n_batches: int, halve_lr: bool = True):
    best_perpl = -1
    max_patience = PATIENCE  # maybe change later
    patience = max_patience

    for i, batch in enumerate(dataloader):
        # set model to training mode to ensure that the gradient tracking is on
        model.train()
        _, _, y = batch
        # loss func needs a row vector converted to long
        y = y.squeeze().to(torch.long)
        y_w = model(batch)
        loss = lossfunc(y_w, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # evaluate every n updates
        if i % eval_every_n_batches == 0:
            eval_perpl, eval_acc = eval_model(model, eval_dataloader, lossfunc, 500)
            print(f"Batch: {i}/{len(dataloader)} Perplexity: {eval_perpl} ,Accuracy {eval_acc}")
            # check if the current perplexity is still better
            if eval_perpl < best_perpl or best_perpl == -1:
                best_perpl = eval_perpl
                patience = max_patience
            else:
                patience -= 1
                if patience <= 0:
                    if halve_lr:
                        # optimizer.param_groups is a list of the different weight groups, but we only have one group
                        optimizer.param_groups[0]["lr"] /= 2
                        print(f"Reducing learning rate to {optimizer.param_groups[0]['lr']}")
                        patience = max_patience
                    else:
                        # stop training if model isn't improving and we shouldn't halve the lr
                        print(
                            f"Stopping training, model not improving. Best perplexity: {best_perpl}, Epoch: {epoch + 1}")
                        break
        if i in checkpoints:
            save_model(model, f"model_epoch_{epoch}_batch_{i}")


def eval_model(model: nn.Module,
               data_loader: DataLoader,
               loss: nn.Module,
               number_of_batches: int):
    model.eval()
    # disable gradient calculation
    total_loss = 0
    num_samples = len(data_loader.dataset) # number of rows
    num_batches = len(data_loader)
    total_acc = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i == number_of_batches:
                break
            _, _, y = batch
            y_w = model(batch)
            # loss func needs a row vector converted to long (it was a column vector)
            y = y.squeeze().to(torch.long)
            total_loss += loss(y_w, y).item()
            """ argmax(1) returns the arg max for each row 
                and y_w.argmax(1) == y returns a boolean tensor and we sum the true values
            """
            total_acc += (y_w.argmax(1) == y).type(torch.float).sum().item()
    # return perplexity and accuracy (e^... since pytorch's cross entropy uses the natural logarithm)
    return np.exp(total_loss/num_batches), (total_acc/num_samples)


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
        if DROPOUT:
            self.drop = nn.Dropout(p=0.1)
        else:
            self.drop = nn.Identity()
        if OWN_LINEAR:
            self.fc_source = FCLayer(embedding_dim * (2 * window_size + 1), FC_OUT_ST)
            self.fc_target = FCLayer(embedding_dim * window_size, self.fc_source.shape_out)
            self.fc_1 = FCLayer(self.fc_source.shape_out + self.fc_target.shape_out, FC_OUT_1)
            self.fc_3 = FCLayer(self.fc_1.shape_out, dict_len, "xavier")
        else:
            self.fc_source = nn.Linear(embedding_dim * (2 * window_size + 1), FC_OUT_ST)
            self.fc_target = nn.Linear(embedding_dim * window_size, self.fc_source.out_features)
            self.fc_1 = nn.Linear(self.fc_source.out_features + self.fc_target.out_features, FC_OUT_1)
            self.fc_3 = nn.Linear(self.fc_1.out_features, dict_len)

            torch.nn.init.kaiming_uniform_(self.fc_source.weight, nonlinearity="relu")
            torch.nn.init.kaiming_uniform_(self.fc_target.weight, nonlinearity="relu")
            torch.nn.init.kaiming_uniform_(self.fc_1.weight, nonlinearity="relu")
            torch.nn.init.xavier_uniform_(self.fc_3.weight)
    def forward(self, item: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        s, t, _ = item
        # get embeddings
        s = self.embedding(s)
        t = self.embedding(t)
        # concatenate the embeddings
        s = torch.flatten(s, start_dim=1)
        t = torch.flatten(t, start_dim=1)
        s = self.drop(torch.relu(self.fc_source(s)))
        t = self.drop(torch.relu(self.fc_target(t)))
        # concatenating s and t
        x = torch.cat((s, t), 1)
        x = self.drop(torch.relu(self.fc_1(x)))
        # no activation
        x = self.fc_3(x)
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
    parser.add_argument("--model", type=str, help="name of model file")
    args = parser.parse_args()

    # loading hyperparameter
    with open('../config/hyperparameter.yml', 'r') as file:
        hyp_param = yaml.safe_load(file)
        EMBEDDING_DIM = hyp_param["EMBEDDING_DIM"]
        EPOCHS = hyp_param["EPOCHS"]
        CHECKPOINTS = [hyp_param["CHECKPOINTS_WBATCH"] if i in hyp_param["CHECKPOINTS_WEPOCH"] else [] for i in range(EPOCHS)]
        LEARNING_RATE = hyp_param["LEARNING_RATE"]
        EVAL_EVERY_N_BATCHES = hyp_param["EVAL_EVERY_N_BATCHES"]
        OWN_LINEAR = hyp_param["OWN_LINEAR"]
        WINDOW_SIZE = hyp_param["WINDOW_SIZE"]
        FC_OUT_ST = hyp_param["FC_OUT_ST"]
        FC_OUT_1 = hyp_param["FC_OUT_1"]
        BATCH_SIZE_TRAIN = hyp_param["BATCH_SIZE_TRAIN"]
        BATCH_SIZE_EVAL = hyp_param["BATCH_SIZE_EVAL"]
        BPE_MERGES = hyp_param["BPE_MERGES"]
        PATIENCE = hyp_param["PATIENCE"]
        DROPOUT = hyp_param["DROPOUT"]
        HALVE_LR = hyp_param["HALVE_LR"]

    # load or create BPE
    if args.bpe is not None:
        bpe = load_bpe(args.bpe)
    else:
        bpe = save_bpe(args.train_de, args.train_en, "bpe.pkl", BPE_MERGES)

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
        train_batch = save_batch(bpe, voc, args.train_de, args.train_en, WINDOW_SIZE, "train_batch.pkl")
        eval_batch = save_batch(bpe, voc, args.eval_de, args.eval_en, WINDOW_SIZE, "eval_batch.pkl")

    # create datasets
    train_dataset = TranslationDataset(train_batch.batches)
    eval_dataset = TranslationDataset(eval_batch.batches)

    # create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False)
    if debug > 0:
        print("start training...")
    # create model
    if args.model is not None:
        model = load_model(args.model)
    else:
        model = FFTranslator(train_batch.voc_len, train_batch.windowsize, EMBEDDING_DIM)
    # training
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    epochs = EPOCHS
    checkpoints = CHECKPOINTS
    training_routine(model, train_dataloader, eval_dataloader, optimizer, epochs, HALVE_LR, checkpoints, EVAL_EVERY_N_BATCHES)
