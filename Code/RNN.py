import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import yaml
from pathlib import Path

from Blatt3 import FCLayer
from Blatt2 import BPE, Dictionary, Batch, FileHandler as fh2
from Blatt1 import FileHandler as fh1, Korpus
from Model import Loader

debug_level = 0

EN_HIDDEN_SIZE = 256
DEC_HIDDEN_SIZE = 256
DEC_LSTM_SIZE = 256
ATTN_SIZE = 256
PATIENCE = 5
BPE_MERGES = 7000
IGNORE_INDEX = 3

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

def bpe2(text: list[list[str]], voc: Dictionary, bpe: BPE) -> list[list[int]]:
    text = bpe.applyBPE(text)
    text2 = voc.filterInput(text)
    copy = []
    for i in range(len(text2)):
        copy.append([])
        copy[i].append(0)
        for j in range(len(text2[i])):
            if text2[i][j] not in voc.getDictionary():
                raise ValueError(f"Word {text2[i][j]} not in dictionary")
            copy[i].append(voc.getDictionary()[text2[i][j]])
        copy[i].append(1)
    return copy


def pad(batch):
    (source_seqs, target_seqs) = zip(*batch)

    # remember lengths for packing later
    source_lengths = torch.Tensor([len(seq) for seq in source_seqs]).to(torch.int64)
    target_lengths = torch.Tensor([len(seq)-1 for seq in target_seqs]).to(torch.int64) # !!! -1 bc eos is removed in training
    source_tens = [torch.Tensor(seq).to(torch.int32) for seq in source_seqs]
    target_tens = [torch.Tensor(seq).to(torch.int32) for seq in target_seqs]
    source_padded = pad_sequence(source_tens, batch_first=True, padding_value=IGNORE_INDEX)
    target_padded = pad_sequence(target_tens, batch_first=True, padding_value=IGNORE_INDEX)

    return source_padded, target_padded, source_lengths, target_lengths


class RNNEncoder(nn.Module):
    def __init__(self, dict_len: int, embedding_size: int, dropout=0.1, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(dict_len, embedding_size)
        self.lstm = nn.LSTM(embedding_size, EN_HIDDEN_SIZE, dropout=dropout, bidirectional=True, num_layers=num_layers, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # x is a batch of sequences
        # x shape: (batch_size, seq_len)
        x = self.drop(self.embedding(x))
        # x shape: (batch_size, seq_len, embedding_size)
        x_pack = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, (h, c) = self.lstm(x_pack)
        x, _ = pad_packed_sequence(x, batch_first=True) # unpack (back to padded), we don't need the lengths
        # x shape: (batchsize, seqsize, 2*en_hidden) (bc biderectional)
        return x


class Attention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(Attention, self).__init__()
        self.A = nn.Linear(encoder_hidden_size, ATTN_SIZE)
        self.B = nn.Linear(decoder_hidden_size, ATTN_SIZE)
        self.V = nn.Linear(ATTN_SIZE, 1)
        self.b_small = nn.Parameter(torch.zeros(ATTN_SIZE))

    def forward(self, f_j, s_i, mask_f_j, mask_s_i):
        #f_j: (batch_size, seq_len, 2*hidden_size) because of bidirectional
        #s_i: (batch_size, hidden_size)
        # s_i: (batch_size, 1, hidden_size)
        # mask_f_j = (batch_size, seq_len_src) type bool
        # mask_s_i = (batch_size, seq_len_tgt) type bool
        s_i = s_i.unsqueeze(2)
        f_j = f_j.unsqueeze(1)
        #print(s_i.shape)
        #print(f_j.shape)
        scores = self.V(torch.tanh(self.A(f_j) + self.B(s_i) + self.b_small))
        # scores: (batch_size, seq_len, 1)
        scores = scores.squeeze(3)
        # scores: (batch_size, seq_len_src, seq_len_targ)
        # unsqueeze for outer product (scores * f_j) which equals the sum over the seq_len
        mask = mask_f_j.unsqueeze(1) + mask_s_i.unsqueeze(2)
        scores.masked_fill_(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # since we want softmax over \alpha_i which is the seq_len
        weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)  # 0/0 is supposed to be 0 in our case
        #print(weights.shape)
        f_j = f_j.squeeze(1)
        #print(f_j.shape)
        context = torch.bmm(weights, f_j)
        #context = torch.sum(weights * f_j, dim=1)
        #print(context.shape)
        return context.squeeze(1)  # 2*EN_HIDDEN_SIZE


    def forward_single(self, f_j, s_i, f_j_mask):
        #f_j: (batch_size, seq_len, 2*hidden_size) because of bidirectional
        #s_i: (batch_size, hidden_size)
        #s_i = s_i.unsqueeze(1)
        # s_i: (batch_size, 1, hidden_size)
        # mask_f_j = (batch_size, seq_len_src) type bool
        scores = self.V(torch.tanh(self.A(f_j) + self.B(s_i) + self.b_small))
        # scores: (batch_size, seq_len, 1)
        scores = scores.squeeze(2).unsqueeze(1)
        # scores: (batch_size, 1, seq_len)
        #mask = f_j_mask.unsqueeze(1)
        #scores.masked_fill_(mask, float("-inf"))
        # unsqueeze for outer product (scores * f_j) which equals the sum over the seq_len
        weights = torch.softmax(scores, dim=-1)  # since we want softmax over \alpha_i which is the seq_len
        context = torch.bmm(weights, f_j)
        #context = torch.sum(weights * f_j, dim=1)
        return context.squeeze(1)  # 2*EN_HIDDEN_SIZE

class RNNDecoder(nn.Module):
    def __init__(self, dict_len: int, embedding_size: int, dropout=0.1, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(dict_len, embedding_size).to(device)
        self.lstm = nn.LSTM(embedding_size, DEC_HIDDEN_SIZE, dropout=dropout, num_layers=num_layers).to(device)
        self.attention = Attention(2 * EN_HIDDEN_SIZE,DEC_HIDDEN_SIZE).to(device)
        self.out = nn.Linear(2 * EN_HIDDEN_SIZE + DEC_HIDDEN_SIZE, dict_len).to(device)

    def forward(self, encoder_outputs, lengths, target_tensor=None):
        """
        @param target_tensor: is used when we want the model to use teacher forcing
                i.e. we feed the correct target word into the decoder and not the predicted one
        """
        batch_size = encoder_outputs.shape[0]
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device)
        decoder_input.fill_(0)  # first input is always <s>
        decoder_hidden_1 = None
        decoder_outputs = []
        max_length = max(lengths)
        for i in range(max_length):
            output, decoder_hidden_1 = self.forward_step(decoder_input, decoder_hidden_1, encoder_outputs, lengths)
            decoder_outputs.append(output)
            if target_tensor is None:
                _, x = output.topk(1)
                decoder_input = x.squeeze(-1).detach()  # check if squeeze is correct
            else:
                decoder_input = target_tensor[:, i].unsqueeze(1)
        #return decoder_outputs
        seq_out = torch.cat(decoder_outputs, dim=1)
        return seq_out

    def forward_beam_search(self, encoder_outputs, lengths, beam_size: int, voc: Dictionary, max_length=None, return_topk = False):
        batch_size = encoder_outputs.shape[0]
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device)
        decoder_input.fill_(0)  # first input is always <s>
        mapping = voc.getDictionary()
        beam = [([mapping["<s>"]], 0.0, None)]  # (sequence, probability, hidden state)
        finished = []
        if max_length is None:
            max_length = max(lengths)
        with torch.no_grad():
            for _ in range(max_length):
                temp = []
                for seq, prob, hidden_state in beam:
                    decoder_input = torch.Tensor([seq[-1]]).unsqueeze(0)
                    output, new_hidden = self.forward_step(decoder_input, hidden_state, encoder_outputs, lengths)
                    output = torch.softmax(output, dim=-1)
                    values, indices = torch.topk(output, beam_size)
                    for val, ind in zip(values.squeeze(), indices.squeeze()):
                        if ind.item() == mapping["</s>"]:
                            finished.append((seq, prob + np.log(val.item()), new_hidden))
                        else:
                            temp.append((seq + [ind.item()], prob + np.log(val.item()), new_hidden))
                beam = sorted(temp, key=lambda x: x[1], reverse=True)[:beam_size]

                # if all sequences have ended, we can stop
                if not beam:
                    break
        res = finished + beam
        if return_topk:
            return sorted(res, key=lambda x: x[1], reverse=True)[:beam_size]
        else:
            return sorted(res, key=lambda x: x[1], reverse=True)[0]


    def forward_step(self, e_prev, s1_prev, f_j, lengths):
        """
        @param e_prev: the previous word (batch_size)
        @param s1_prev: the previous hidden state; 2-tuple of (h, c)?
        @param f_j: the encoder outputs (batch_size, seq_len, 2*hidden_size)
        """
        e_prev = e_prev.to(torch.long)
        embedded = self.embedding(e_prev)  # maybe add dropout later
        # embedded: (batch_size, 1, embedding_size)
        if s1_prev is None:
            e, s1 = self.lstm(embedded)
        else:
            e, s1 = self.lstm(embedded, s1_prev)
        # (batch_size, 1, hidden_size), 2-tuple of (h, c)
        e = e.squeeze(1)
        # (batch_size, hidden_size)
        attention_context = self.attention.forward_single(f_j, s1[0])
        input_lstm = torch.cat((attention_context, e), dim=1)
        output = self.out(input_lstm)
        #output = pad_packed_sequence(output, batch_first=True)
        return output.unsqueeze(1), s1  # unsqueeze so we can concatenate the outputs later


class RNNDecoder_unrolled(nn.Module):
    def __init__(self, dict_len: int, embedding_size: int, dropout=0.1, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(dict_len, embedding_size)
        self.lstm = nn.LSTM(embedding_size, DEC_HIDDEN_SIZE, dropout=dropout, num_layers=num_layers, batch_first=True)
        self.attention = Attention(2*EN_HIDDEN_SIZE,DEC_HIDDEN_SIZE)
        self.out = nn.Linear(2 * EN_HIDDEN_SIZE + DEC_HIDDEN_SIZE, dict_len)
        self.drop = nn.Dropout(dropout)

    def forward(self, input, targ_lengths, f_j, mask_f_j, mask_s_i):

        """
        @param input: the whole sequence
        @param targ_lengths: the lengths of the sequences
        @param f_j: the encoder outputs
        """
        embedded = self.drop(self.embedding(input))  # maybe add dropout later
        # embedded: (batch_size, seq_len, embedding_size)
        x_pack = pack_padded_sequence(embedded, targ_lengths, batch_first=True, enforce_sorted=False)
        x, (h, c) = self.lstm(x_pack)
        x, _ = pad_packed_sequence(x, batch_first=True)
        # (batch_size, 1, hidden_size), 2-tuple of (h, c)
        attention_context = self.attention(f_j, x, mask_f_j, mask_s_i)
        linear_input = torch.cat((attention_context, x), dim=2)
        output = self.out(linear_input)

        return output  # unsqueeze so we can concatenate the outputs later

    def forward_beam_search(self, encoder_outputs, f_j_mask, lengths, beam_size: int, voc: Dictionary, max_length=None, return_topk = False):
        batch_size = encoder_outputs.shape[0]
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device)
        decoder_input.fill_(0).to(device)  # first input is always <s>
        mapping = voc.getDictionary()
        beam = [([mapping["<s>"]], 0.0, None)]  # (sequence, probability, hidden state)
        finished = []
        if max_length is None:
            max_length = max(lengths)
        with torch.no_grad():
            for _ in range(max_length):
                temp = []
                for seq, prob, hidden_state in beam:
                    decoder_input = torch.Tensor([seq[-1]]).unsqueeze(0).to(torch.long).to(device)
                    output, new_hidden = self.forward_step(decoder_input, hidden_state, encoder_outputs, f_j_mask, lengths)
                    output = torch.softmax(output, dim=-1)
                    values, indices = torch.topk(output, beam_size)
                    for val, ind in zip(values.squeeze(), indices.squeeze()):
                        if ind.item() == mapping["</s>"]:
                            new_prob = prob + np.log(val.item())
                            # normalize by length
                            new_prob = new_prob / len(seq)
                            finished.append((seq, new_prob, new_hidden))
                        else:
                            temp.append((seq + [ind.item()], prob + np.log(val.item()), new_hidden))
                beam = sorted(temp, key=lambda x: x[1], reverse=True)[:beam_size]

                # if all sequences have ended, we can stop
                if not beam:
                    break
        res = finished + beam
        # remove the eos token
        if return_topk:
            return sorted(res, key=lambda x: x[1], reverse=True)[:beam_size]
        else:
            return sorted(res, key=lambda x: x[1], reverse=True)[0]


    def forward_step(self, e_prev, s1_prev, f_j, f_j_mask, lengths):
        """
        @param e_prev: the previous word (batch_size)
        @param s1_prev: the previous hidden state; 2-tuple of (h, c)?
        @param f_j: the encoder outputs (batch_size, seq_len, 2*hidden_size)
        """
        embedded = self.embedding(e_prev)  # maybe add dropout later
        # embedded: (batch_size, 1, embedding_size)
        #print(f"Embedded shape: {embedded.shape}")
        #print(f"Lengths: {lengths}")
        #embedded = pack_padded_sequence(embedded, torch.Tensor(lengths), batch_first=True, enforce_sorted=False)
        #print(f"Embedded shape after packing: {embedded.data.shape}")
        if s1_prev is None:
            #print("s1_prev is None")
            e, s1 = self.lstm(embedded)
        else:
            #print(f"s1_prev[0] shape: {s1_prev[0].shape}")
            #s1_prev = (s1_prev[0].squeeze(1), s1_prev[0].squeeze(1))
            e, s1 = self.lstm(embedded, s1_prev)
        # (batch_size, 1, hidden_size), 2-tuple of (h, c)
        e = e.squeeze(1)
        # (batch_size, hidden_size)
        attention_context = self.attention.forward_single(f_j, s1[0], f_j_mask)
        #attention_context = attention_context.squeeze(1)
        input_lstm = torch.cat((attention_context, e), dim=1)
        """
        if s2_prev is None:
            e, s2 = self.lstm2(input_lstm)
        else:
            e, s2 = self.lstm2(input_lstm, s2_prev)
        """
        output = self.out(input_lstm)
        #output = pad_packed_sequence(output, batch_first=True)
        return output.unsqueeze(1), s1  # unsqueeze so we can concatenate the outputs later

def eval_model(encoder: nn.Module,
               decoder: nn.Module,
               data_loader: DataLoader,
               loss: nn.Module,
               number_of_batches: int):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    num_samples = len(data_loader.dataset)  # number of rows
    total_acc = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i == number_of_batches:
                break
            source_pad, targ_pad, source_len, targ_len = batch
            # loss func needs a row vector converted to long
            y = targ_pad.squeeze().to(torch.long)
            y_enc = encoder(source_pad, source_len)
            y_dec = decoder(y_enc, targ_len)  # dim: (batch_size, seq_len, vocab_size)
            y_dec = torch.permute(y_dec, (0, 2, 1))  # dim: (batch_size, vocab_size, seq_len) needed for cross entropy
            total_loss += loss(y_dec, y).item()
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
        source_mask = (source_pad == IGNORE_INDEX).to(device)
        targ_mask = (targ_pad == IGNORE_INDEX).to(device)
        # print(t.shape, s.shape, y.shape)
        # loss func needs a row vector converted to long
        source_pad = source_pad.to(device)
        y_enc = encoder(source_pad, source_len)
        y_enc = y_enc.to(device)
        y_in_dec = targ_pad.squeeze().to(device)[:,:-1] # removing the eos
        targ_mask = targ_mask[:,:-1]
        for j in range(len(targ_mask)):
            if sum(targ_mask[j]) + 1 >= len(targ_mask[j]):
                print('found')
        y_dec = decoder(y_in_dec, targ_len, y_enc, source_mask, targ_mask) # dim: (batch_size, seq_len, vocab_size)
        y_dec = torch.permute(y_dec, (0, 2, 1))  # dim: (batch_size, vocab_size, seq_len) needed for cross entropy
        #for j in range(len(y_dec)):
            #debug = y[:, j]
            #loss+= lossfunc(y_dec[j], y[:, j])
        y = targ_pad.squeeze().to(torch.long).to(device)[:,1:]  # removing the sos
        loss = lossfunc(y_dec, y) #/ (y.shape[0]*y.shape[1])  # divide by batch size bc we used summed loss
        if i % 50 == 0:
          print(f"current Loss: {loss}")
        loss.backward()
        optimizer_enc.step()
        optimizer_enc.zero_grad()
        optimizer_dec.step()
        optimizer_dec.zero_grad()
        if i in checkpoints:
            Loader.save_model(encoder, f"encoder_epoch_{epoch + 1}_batch_{i}")
            Loader.save_model(decoder, f"decoder_epoch_{epoch + 1}_batch_{i}")

def training_routine(encoder: nn.Module,
                     decoder: nn.Module,
                     train_dataloader: DataLoader,
                     eval_dataloader: DataLoader,
                     optimizer_enc: torch.optim,
                     optimizer_dec: torch.optim,
                     epochs: int,
                     halve_lr: bool,
                     checkpoints: list[list[int]],
                     eval_every_n_batches: int):
    loss_func = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    for name, param in encoder.named_parameters():
        print(f"Parameter name: {name}, Shape: {param.shape}")
    for name, param in decoder.named_parameters():
        print(f"Parameter name: {name}, Shape: {param.shape}")
    n = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in encoder.parameters())
    print(f"Number of total parameters: {n}")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        train_one_epoch(encoder, decoder, train_dataloader, eval_dataloader, optimizer_enc, optimizer_dec, checkpoints[epoch + 1],
                        loss_func, epoch,
                        eval_every_n_batches, halve_lr)

def beam_wrapper(encoder: nn.Module, decoder: nn.Module, dataloader, beam_size: int, voc: Dictionary):
    debug = 0
    reverse_dict = {v: k for k, v in voc.getDictionary().items()}
    encoder.eval()
    decoder.eval()
    reverse_voc = {v: k for k, v in voc.getDictionary().items()}
    result = []
    for batch in dataloader:
        source_pad, targ_pad, source_len, targ_len = batch
        source_pad = source_pad.to(device)
        source_mask = (source_pad == IGNORE_INDEX).to(device)
        for i in range(len(source_pad)):
            y_enc = encoder(source_pad[i].unsqueeze(0), source_len[i].unsqueeze(0))
            y_dec = decoder.forward_beam_search(y_enc.to(device), source_mask[i].unsqueeze(0), targ_len.to(device), beam_size, voc)
            if debug > 0:
                for i in source_pad[i]:
                    word = i.item()
                    print(reverse_voc[word], end=" ")
                for i in y_dec[0]:
                    print(reverse_voc[i], end=" ")
            mark = False
            newelem = []
            y_dec = y_dec[0]
            y_dec = [reverse_dict[i] for i in y_dec]
            for word in y_dec:
                if mark:
                    newelem[-1] = newelem[-1][:-2] + word  # remove @@ from the last element and merge
                    mark = False
                else:
                    newelem.append(word)
                if word[-2:] == "@@":
                    mark = True
            y_dec = newelem
            #print(y_dec)
            result.append(y_dec[1:])
        return result
# Extract the epoch and batch numbers from the model name
def extract_numbers(name):
    parts = name.split('_')
    epoch_number = int(parts[2])
    batch_number = int(parts[4])
    return (epoch_number, batch_number)



def calc_bleus(model_path, dataloader, bpe, voc, beam_size: int):
    # listing all modelnames in the directory
    target_eval = fh1.read_file_words(args.eval_en)
    path = Path(model_path)
    decoder_model_names = [file.name for file in path.rglob('*') if file.is_file() and not file.name.startswith(".") and not file.name.startswith("encode")]
    encoder_model_names = [file.name for file in path.rglob('*') if file.is_file() and not file.name.startswith(".") and not file.name.startswith("decode")]
    # Sort the model names based on the epoch and batch numbers
    encoder_model_names = sorted(encoder_model_names, key=extract_numbers)
    decoder_model_names = sorted(decoder_model_names, key=extract_numbers)
    print("Models loaded")
    # list of bleu_scores archieved with the different models
    bleu_list = []
    for enc_name, dec_name in zip(encoder_model_names, decoder_model_names):
        encoder = Loader.load_model(enc_name)
        decoder = Loader.load_model(dec_name)
        print(f"Model: {dec_name}, {enc_name}")
        # translating setences one by one
        hyp = beam_wrapper(encoder, decoder, dataloader, beam_size, voc)
        korp = Korpus(target_eval, hyp)
        print(f"BLEU: {korp.bleu4()}")
        bleu_list.append((korp.bleu4(), dec_name, enc_name))
    return bleu_list



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_de", type=str, help="path to de training data")
    parser.add_argument("--train_en", type=str, help="path to en training data")
    parser.add_argument("--eval_de", type=str, help="path to de evaluation data")
    parser.add_argument("--eval_en", type=str, help="path to en evaluation data")
    parser.add_argument("--bpe", type=str, help="name of bpe file")
    parser.add_argument("--voc", type=str, help="name of voc file")
    parser.add_argument("--train_dataset", type=str, help="path to de training data")
    parser.add_argument("--eval_dataset", type=str, help="path to en training data")
    parser.add_argument("--should_train", action="store_true", help="train the model?")
    args = parser.parse_args()

    # load or create BPE
    if args.bpe is not None:
        print("Loading BPE")
        bpe = Loader.load_bpe(args.bpe)
    else:
        print("Creating BPE")
        bpe = Loader.save_bpe(args.train_de, args.train_en, "bpe.pkl", BPE_MERGES)

    # load or create vocabulary
    if args.voc is not None:
        print("Loading Vocabulary")
        voc = Loader.load_voc(args.voc)
    else:
        print("Creating Vocabulary")
        voc = Loader.save_voc(args.train_de, args.train_en, "voc.pkl", bpe)

    # create dataset
    if args.train_dataset and args.eval_dataset is not None:
        dataset = torch.load(args.train_dataset)
        dataseteval = torch.load(args.eval_dataset)
    else:
        print("Creating dataset")
        train_de = fh1.read_file_words(args.train_de)
        train_en = fh1.read_file_words(args.train_en)
        train_de = bpe2(train_de, voc, bpe)
        train_en = bpe2(train_en, voc, bpe)
        eval_de = fh1.read_file_words(args.eval_de)
        eval_en = fh1.read_file_words(args.eval_en)
        eval_de = bpe2(eval_de, voc, bpe)
        eval_en = bpe2(eval_en, voc, bpe)
        dataset = RNNDataSet(train_de, train_en)
        torch.save(dataset, "../config/dataset_t.pkl")
        dataseteval = RNNDataSet(eval_de, eval_en)
        torch.save(dataseteval, "../config/dataset_e.pkl")
    device = "cpu"
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")

    else:
        print("MPS is available.")
        device = torch.device("cpu")
    data_loader = DataLoader(dataset=dataset, batch_size=256, shuffle=True, collate_fn=pad)
    data_loader_eval = DataLoader(dataset=dataseteval, batch_size=256, shuffle=False, collate_fn=pad)
    if args.should_train:
        print("Training model")
        model_enc = RNNEncoder(len(voc.getDictionary()), 256).to(device)
        model_dec = RNNDecoder_unrolled(len(voc.getDictionary()), 256).to(device)
        optimizer_enc = torch.optim.Adam(model_enc.parameters(), lr=0.005)
        optimizer_dec = torch.optim.Adam(model_dec.parameters(), lr=0.005)
        training_routine(model_enc, model_dec, data_loader, data_loader, optimizer_enc, optimizer_dec, 10, True, [[], [],[],[50],[],[],[50],[50],[],[1],[],[],[1]], 10)
    #model_enc = Loader.load_model("encoder3_epoch_6_batch_50").to(device)
    #model_dec = Loader.load_model("decoder3_epoch_6_batch_50").to(device)
    #beam_wrapper(model_enc, model_dec, data_loader_eval, 5, voc)
    print(calc_bleus("../config/Models", data_loader_eval, bpe, voc, 10))


