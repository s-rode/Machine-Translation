import torch
import torch.nn as nn
import torch.nn.functional as F
import Model
import numpy as np
import argparse

from Blatt2 import BPE, Dictionary, Batch, FileHandler
from Blatt1 import FileHandler as fh1
from Model import TranslationDataset, Loader, FFTranslator
from torch.utils.data import DataLoader, Dataset


class Scoring:
    @staticmethod
    def score(model: nn.Module, source_sentences, target_sentences, bpe: BPE, voc: Dictionary):
        #windowsize = int(model.fc_target.weights.shape[0]/model.embedding.weight.shape[1])
        windowsize = model.windowsize
        model.eval()
        with torch.no_grad():
            scores = []
            for source, target in zip(source_sentences, target_sentences):
                dataloader = Scoring.create_sentence_dataloader(source, target, bpe, voc, windowsize)
                batch = next(iter(dataloader))
                _, _, y = batch
                y_w = model(batch)
                score = Scoring.calculate_score(y_w, y)
                scores.append(score)
            return scores

    @staticmethod
    def calculate_score(y_w, y):
        y = y.squeeze().to(torch.long)
        loss = F.cross_entropy(y_w, y, reduction="sum")
        loss = -loss.item() / y.shape[0]
        prob = torch.exp(torch.tensor(loss))
        return prob.item()

    def create_sentence_dataloader(source_sentence: list[str], target_sentence: list[str], bpe, voc, windowsize):
        filtered_source = voc.filterInput(bpe.applyBPE([source_sentence]))
        filtered_target = voc.filterInput(bpe.applyBPE([target_sentence]))

        batch_size = len(filtered_target[0]) + 1  # +1 for </s>
        batch = Batch(filtered_source, filtered_target, batch_size, windowsize, voc, voc)

        data = TranslationDataset(batch.batches)

        return DataLoader(data, batch_size=batch_size, shuffle=False)


class BeamSearch:
    def __init__(self, source_path: str, target_path: str, bpe: BPE, voc: Dictionary):
        self.ratio = calculate_ratio(source_path, target_path)
        self.bpe = bpe
        self.voc = voc

    def beam_search(self, model: nn.Module, beam_size: int, dataloader: DataLoader, voc: Dictionary):
        mapping = voc.getDictionary()
        #results = [[[mapping["<s>"]], np.log(1)] for _ in range(beam_size)]
        results = [[[mapping["<s>"]], 0]]

        mapping = voc.getDictionary()
        reverse_mapping = {v: k for k, v in mapping.items()}  # Create a reverse mapping

        for batch in dataloader:
            x, _, blank = batch
            temp = []
            for elem in results:
                last_ws = elem[0][-model.windowsize:]  # last window size elements
                if len(last_ws) < model.windowsize:  # padding
                    last_ws = [mapping["<s>"]] * (model.windowsize - len(last_ws)) + last_ws
                out = model((x, torch.tensor(last_ws).unsqueeze(0), blank))
                out = F.softmax(out, dim=1)
                # get the best beam_size elements
                (values, indices) = torch.topk(out, beam_size)
                # todo: </s> handling
                # values is column vector so (1, beam_size)
                for k in range(values.shape[1]): # iterate over beam_size
                    temp.append([elem[0] + [indices[0, k].item()], elem[1] + np.log(values[0, k].item())])
                # convert back to string for debugging
                string_sequences = [[reverse_mapping.get(token, '<unk>') for token in sequence] for sequence, _ in temp]

            results = sorted(temp, key=lambda x: x[1], reverse=True)[:beam_size] # get the best beam_size elements
        return results

    def beam_wrapper(self, model: nn.Module, source_sentence: list[str], beam_size: int, offset: int):
        filtered_source = self.voc.filterInput(self.bpe.applyBPE([source_sentence]))
        pred_len = round(len(filtered_source[0]) * self.ratio)
        results = []

        mapping = self.voc.getDictionary()
        reverse_mapping = {v: k for k, v in mapping.items()}  # Create a reverse mapping for token indices to tokens

        for length in range(pred_len - offset, pred_len + offset + 1):
            padding = [["<s>"] * length]
            # todo: check batch (debug)
            batch = Batch(filtered_source, padding, length + 1, model.windowsize, self.voc, self.voc)
            data = TranslationDataset(batch.batches)
            dataloader = DataLoader(data, batch_size=1, shuffle=False)
            res = self.beam_search(model, beam_size, dataloader, self.voc)
            # concat
            results += res

        results = sorted(results, key=lambda x: x[1], reverse=True)[:beam_size]
        # convert back to string
        results_str = []
        for sequence, prob in results:
            print(prob)
            for token in sequence:
                print(reverse_mapping.get(token, '<unk'), end=" ")


def calculate_ratio(source_path: str, target_path: str):
    source = fh1.read_file_words(source_path)
    target = fh1.read_file_words(target_path)

    source_avg = sum([len(s) for s in source]) / len(source)
    target_avg = sum([len(t) for t in target]) / len(target)

    return target_avg / source_avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_de", type=str, help="path to de training data")
    parser.add_argument("--train_en", type=str, help="path to en training data")
    parser.add_argument("--eval_de", type=str, help="path to de evaluation data")
    parser.add_argument("--eval_en", type=str, help="path to en evaluation data")
    parser.add_argument("--bpe", type=str, help="name of bpe file")
    parser.add_argument("--voc", type=str, help="name of voc file")
    parser.add_argument("--model", type=str, help="name of model file")
    parser.add_argument("--beam_size", type=int, help="size of beam")
    parser.add_argument("--offset", type=int, help="offset for beam search")
    args = parser.parse_args()

    bpe = Loader.load_bpe(args.bpe)
    voc = Loader.load_voc(args.voc)
    model = Loader.load_model(args.model)
    bs = BeamSearch(args.train_de, args.train_en, bpe, voc)
    source = fh1.read_file_words(args.eval_de)
    #s1 = source[3]
    s1 = ["drei", "alte", "männer", "sehen", "einem", "anderen", "mann", "zu", ",", "wie", "er", "fisch", "zubereitet", "."]
    bs.beam_wrapper(model, s1, args.beam_size, args.offset)


