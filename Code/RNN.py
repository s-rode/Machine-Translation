import argparse

import torch
import torch.nn as nn
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import yaml

from Blatt3 import FCLayer
from Blatt2 import BPE, Dictionary, Batch, FileHandler as fh2
from Blatt1 import FileHandler as fh1
from Model import Loader

EN_HIDDEN_SIZE = 256
DEC_HIDDEN_SIZE = 256
DEC_LSTM_SIZE = 256
ATTN_SIZE = 256
PATIENCE = 5
BPE_MERGES = 7000


class RNNDataSet(Dataset):
    def __init__(self, source: list[list[int]], target: list[list[int]]):
        # big_batch has only a single element
        self.source = source
        self.target = target

    def __len__(self):
        # since each matrix has the same number of rows
        return len(self.source)

    def __getitem__(self, index):
        return self.source[index], self.target[index]


def bpe_and_to_int(text: list[list[str]], voc: Dictionary, bpe: BPE) -> list[list[int]]:
    text = bpe.applyBPE(text)
    copy = []
    for i in range(len(text)):
        copy.append([])
        for j in range(len(text[i])):
            if text[i][j] not in voc.getDictionary():
                if text[i][j][-2:] == "@@":
                    copy[i].append(3)  # append <unk>@@
                else:
                    copy[i].append(2)  # append <unk>
            else:
                copy[i].append(voc.getDictionary()[text[i][j]])
        copy[i].append(1)  # append </s>
    return copy


def pad(batch):
    (source_seqs, target_seqs) = zip(*batch)

    # remember lengths for packing later
    source_lengths = [len(seq) for seq in source_seqs]
    target_lengths = [len(seq) for seq in target_seqs]
    source_tens = [torch.Tensor(seq) for seq in source_seqs]
    target_tens = [torch.Tensor(seq) for seq in target_seqs]
    source_padded = pad_sequence(source_tens, batch_first=True, padding_value=-1)
    target_padded = pad_sequence(target_tens, batch_first=True, padding_value=-1)

    return source_padded, target_padded, source_lengths, target_lengths


class RNNEncoder(nn.Module):
    def __init__(self, dict_len: int, embedding_size: int, dropout=0.1, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(dict_len, embedding_size)
        self.lstm = nn.LSTM(embedding_size, EN_HIDDEN_SIZE, dropout=dropout, bidirectional=True, num_layers=num_layers)

    def forward(self, x: torch.Tensor, lengths: list[int]):
        # x is a batch of sequences
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)
        # x shape: (batch_size, seq_len, embedding_size)
        x = pack_padded_sequence(x, torch.Tensor(lengths), batch_first=True, enforce_sorted=False)
        """
        if we have only one layer the last entry of x is h
        """
        x, (h, c) = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)  # unpack (back to padded), we don't need the lengths

        # we don't need the hidden state
        # x shape: (batchsize, seqsize, 2*en_hidden) (bc biderectional)
        return x


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.A = nn.Linear(2 * EN_HIDDEN_SIZE, ATTN_SIZE)
        self.B = nn.Linear(hidden_size, ATTN_SIZE)
        self.V = nn.Linear(ATTN_SIZE, 1)
        self.b_small = nn.Parameter(torch.zeros(ATTN_SIZE))

    def forward(self, f_j, s_i):
        """
        f_j: (batch_size, seq_len, 2*hidden_size) because of bidirectional
        s_i: (batch_size, hidden_size)
        """
        s_i = s_i.unsqueeze(1)
        # s_i: (batch_size, 1, hidden_size)
        scores = self.V(torch.tanh(self.A(f_j) + self.B(s_i) + self.b_small))
        # scores: (batch_size, seq_len, 1)
        scores = scores.squeeze(2).unsqueeze(1)
        # scores: (batch_size, 1, seq_len)
        # unsqueeze for outer product (scores * f_j) which equals the sum over the seq_len
        weights = torch.softmax(scores, dim=-1)  # since we want softmax over \alpha_i which is the seq_len
        context = torch.bmm(weights, f_j)
        return context  # 2*EN_HIDDEN_SIZE


class RNNDecoder(nn.Module):
    def __init__(self, dict_len: int, embedding_size: int, dropout=0.1, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(dict_len, embedding_size)
        self.lstm = nn.LSTM(embedding_size, DEC_HIDDEN_SIZE, dropout=dropout, num_layers=num_layers)
        self.attention = Attention(DEC_HIDDEN_SIZE)
        self.lstm2 = nn.LSTM(2 * EN_HIDDEN_SIZE + DEC_HIDDEN_SIZE, DEC_LSTM_SIZE, dropout=dropout,
                             num_layers=num_layers)
        self.out = nn.Linear(DEC_LSTM_SIZE, dict_len)

    def forward(self, encoder_outputs, lengths, target_tensor=None):
        """
        @param target_tensor: is used when we want the model to use teacher forcing
                i.e. we feed the correct target word into the decoder and not the predicted one
        """
        batch_size = encoder_outputs.shape[0]
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long)
        decoder_input.fill_(0)  # first input is always <s>
        decoder_hidden_1 = None
        decoder_hidden_2 = None
        decoder_outputs = []
        max_length = max(lengths)
        for i in range(max_length):
            output, decoder_hidden_1, decoder_hidden_2 = self.forward_step(decoder_input, decoder_hidden_1,
                                                                           decoder_hidden_2, encoder_outputs, lengths)
            decoder_outputs.append(output)
            if target_tensor is None:
                _, x = output.topk(1)
                decoder_input = x.squeeze().detach()  # check if squeeze is correct
            else:
                decoder_input = target_tensor[:, i].unsqueeze(1)
        return torch.cat(decoder_outputs, dim=1)

    def forward_step(self, e_prev, s1_prev, s2_prev, f_j, lengths):
        """
        @param e_prev: the previous word (batch_size)
        @param s1_prev: the previous hidden state; 2-tuple of (h, c)?
        @param s2_prev: the prev hidden state of the lstm2
        @param f_j: the encoder outputs (batch_size, seq_len, 2*hidden_size)
        """
        embedded = self.embedding(e_prev)  # maybe add dropout later
        # embedded: (batch_size, 1, embedding_size)
        embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        if s1_prev is None:
            e, s1 = self.lstm(embedded)
        else:
            e, s1 = self.lstm(embedded, s1_prev)
        # (batch_size, 1, hidden_size), 2-tuple of (h, c)
        e = e.squeeze(1)
        # (batch_size, hidden_size)
        attention_context = self.attention(f_j, e)
        input_lstm = torch.cat((attention_context, e), dim=1)
        if s2_prev is None:
            e, s2 = self.lstm2(input_lstm)
        else:
            e, s2 = self.lstm2(input_lstm, s2_prev)
        output = self.out(e)
        output = pad_packed_sequence(output, batch_first=True)
        return output, s1, s2


def eval_model(encoder: nn.Module,
               decoder: nn.Module,
               data_loader: DataLoader,
               loss: nn.Module,
               number_of_batches: int):
    encoder.eval()
    decoder.eval()
    # disable gradient calculation
    total_loss = 0
    num_samples = len(data_loader.dataset)  # number of rows
    total_acc = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i == number_of_batches:
                break

            source_pad, targ_pad, source_len, targ_len = batch
            # print(t.shape, s.shape, y.shape)
            # loss func needs a row vector converted to long
            y = targ_pad.squeeze().to(torch.long)
            y_enc = encoder(source_pad, source_len)
            y_dec = decoder(y_enc, targ_len)
            # loss func needs a row vector converted to long (it was a column vector)
            y = y.squeeze().to(torch.long)
            total_loss += loss(y_dec, y).item()
            total_acc += (y_dec.argmax(1) == y).type(torch.float).sum().item()  # todo: correct????
    # return perplexity and accuracy (e^... since pytorch's cross entropy uses the natural logarithm)
    return np.exp(total_loss / num_samples), (total_acc / num_samples)


def train_one_epoch(encoder: nn.Module,
                    decoder: nn.Module,
                    dataloader: DataLoader,
                    eval_dataloader: DataLoader,
                    optimizer_enc: torch.optim,
                    optimizer_dec: torch.optim,
                    checkpoints: list[int],
                    lossfunc: nn.Module, epoch: int, eval_every_n_batches: int, halve_lr: bool = True):
    best_perpl = -1
    max_patience = PATIENCE  # maybe change later
    patience = max_patience
    for i, batch in enumerate(dataloader):
        # set model to training mode to ensure that the gradient tracking is on
        encoder.train()
        decoder.train()
        source_pad, targ_pad, source_len, targ_len = batch
        # print(t.shape, s.shape, y.shape)
        # loss func needs a row vector converted to long
        y = targ_pad.squeeze().to(torch.long)
        y_enc = encoder(source_pad, source_len)
        y_dec = decoder(y_enc, targ_len)
        loss = lossfunc(y_dec, y) / y.shape[0]  # divide by batch size cause we used summed loss
        loss.backward()
        optimizer_enc.step()
        optimizer_enc.zero_grad()
        optimizer_dec.step()
        optimizer_dec.zero_grad()
        # evaluate every n updates
        if i % eval_every_n_batches == 0:
            eval_perpl, eval_acc = eval_model(encoder, decoder, eval_dataloader, lossfunc, len(eval_dataloader))
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
                        optimizer_dec.param_groups[0]["lr"] /= 2
                        optimizer_enc.param_groups[0]["lr"] /= 2
                        print(
                            f"Reducing learning rate for encoder to {optimizer_enc.param_groups[0]['lr']} and decoder to {optimizer_dec.param_groups[0]['lr']}")
                        patience = max_patience
                    else:
                        # stop training if model isn't improving and we shouldn't halve the lr
                        print(
                            f"Stopping training, model not improving. Best perplexity: {best_perpl}, Epoch: {epoch + 1}")
                        break
        if i in checkpoints:
            Loader.save_model(encoder, f"encoder_epoch_{epoch + 1}_batch_{i}")
            Loader.save_model(decoder, f"decoder_epoch_{epoch + 1}_batch_{i}")


def training_routine(encoder: nn.Module,
                     decoder: nn.Module,
                     train_dataloader: DataLoader,
                     eval_dataloader: DataLoader,
                     optimizer: torch.optim,
                     epochs: int,
                     halve_lr: bool,
                     checkpoints: list[list[int]],
                     eval_every_n_batches: int):
    loss_func = nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)
    for name, param in encoder.named_parameters():
        print(f"Parameter name: {name}, Shape: {param.shape}")
    for name, param in decoder.named_parameters():
        print(f"Parameter name: {name}, Shape: {param.shape}")
    n = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in encoder.parameters())
    print(f"Number of total parameters: {n}")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        train_one_epoch(encoder, decoder, train_dataloader, eval_dataloader, optimizer, checkpoints[epoch + 1],
                        loss_func, epoch,
                        eval_every_n_batches, halve_lr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_de", type=str, help="path to de training data")
    parser.add_argument("--train_en", type=str, help="path to en training data")
    parser.add_argument("--eval_de", type=str, help="path to de evaluation data")
    parser.add_argument("--eval_en", type=str, help="path to en evaluation data")
    parser.add_argument("--bpe", type=str, help="name of bpe file")
    parser.add_argument("--voc", type=str, help="name of voc file")
    parser.add_argument("--train_de_int", type=str, help="path to de training data")
    parser.add_argument("--train_en_int", type=str, help="path to en training data")
    args = parser.parse_args()

    # load or create BPE
    if args.bpe is not None:
        bpe = Loader.load_bpe(args.bpe)
    else:
        bpe = Loader.save_bpe(args.train_de, args.train_en, "bpe.pkl", BPE_MERGES)

    # load or create vocabulary
    if args.voc is not None:
        voc = Loader.load_voc(args.voc)
    else:
        voc = Loader.save_voc(args.train_de, args.train_en, "voc.pkl", bpe)

    # create dataset
    eval_de = fh1.read_file_words(args.eval_de)
    eval_en = fh1.read_file_words(args.eval_en)
    if args.train_de_int is not None:
        train_de = fh1.read_file_ints(args.train_de_int)
        train_en = fh1.read_file_ints(args.train_en_int)
    else:
        train_de = fh1.read_file_words(args.train_de)
        train_en = fh1.read_file_words(args.train_en)
        train_de = bpe_and_to_int(train_de, voc, bpe)
        train_en = bpe_and_to_int(train_en, voc, bpe)
        fh1.write_ints_to_file("train_de_int.txt", train_de)
        fh1.write_ints_to_file("train_en_int.txt", train_en)
    dataset = RNNDataSet(train_de, train_en)
    data_loader = DataLoader(dataset=dataset, batch_size=256, shuffle=True, collate_fn=pad)
    model_enc = RNNDecoder(len(voc.getDictionary()), 256)
    model_dec = RNNEncoder(len(voc.getDictionary()), 256)
    optimizer_enc = torch.optim.Adam(model_enc.parameters(), lr=0.005)
    optimizer_dec = torch.optim.Adam(model_dec.parameters(), lr=0.005)
    training_routine(model_enc, model_dec, data_loader, data_loader, optimizer_enc, 10, True, [[], []], 100)
