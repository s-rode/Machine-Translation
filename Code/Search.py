import torch
import torch.nn as nn
import torch.nn.functional as F
import Model
import numpy as np
import argparse
from pathlib import Path

from preprocessing import BPE, Dictionary, Batch, FileHandler
from translation_metrics import Korpus, FileHandler as fh1
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

    @staticmethod
    def create_sentence_dataloader(source_sentence: list[str], target_sentence: list[str], bpe, voc, windowsize):
        filtered_source = voc.filterInput(bpe.applyBPE([source_sentence]))
        filtered_target = voc.filterInput(bpe.applyBPE([target_sentence]))

        batch_size = len(filtered_target[0]) + 1  # +1 for </s>
        batch = Batch(filtered_source, filtered_target, batch_size, windowsize, voc, voc)

        data = TranslationDataset(batch.batches)

        return DataLoader(data, batch_size=batch_size, shuffle=False)  # dataloader has only one element


class GreedySearch:
    def __init__(self, source_path: str, target_path: str, bpe: BPE, voc: Dictionary):
        self.bpe = bpe
        self.voc = voc
        self.reverse_voc = {v: k for k, v in voc.getDictionary().items()}
        self.ratio = calculate_ratio(source_path, target_path)

    def greedy_search(self, model: nn.Module, dataloader: DataLoader):
        mapping = self.voc.getDictionary()
        results = [mapping["<s>"]]
        for batch in dataloader:  # batchloader has i_max elements
            x, _, blank = batch
            with torch.no_grad():
                last_ws = results[-model.windowsize:]  # last window size elements
                if len(last_ws) < model.windowsize:
                    last_ws = [mapping["<s>"]] * (model.windowsize - len(last_ws)) + last_ws
                model.eval()
                out = model((x, torch.tensor(last_ws).unsqueeze(0), blank))
                out = F.softmax(out, dim=1)  # dim (1,voc_size)
                _, ind = torch.max(out, dim=1)
                results.append(ind.item())
                if ind.item() == mapping["</s>"]:
                    break
                # for debugging
                #print(self.reverse_voc.get(ind.item(), '<unk'), end=" ")
        return results

    def greedy_wrapper(self, model: nn.Module, source_sentence: list[str], i_max: int, revert_bpe: bool = True):
        filtered_source = self.voc.filterInput(self.bpe.applyBPE([source_sentence]))  # filtered_source has only one element
        if i_max == 0:
            i_max = round(len(filtered_source[0]) * self.ratio)
        batchsize = i_max
        padding = [["<s>"] * batchsize]
        batch = Batch(filtered_source, padding, batchsize + 1, model.windowsize, self.voc, self.voc)
        data = TranslationDataset(batch.batches)
        dataloader = DataLoader(data, batch_size=1, shuffle=False)
        greedy_res = self.greedy_search(model, dataloader)
        # convert back to string
        greedy_res_str = [self.reverse_voc.get(token, '<unk>') for token in greedy_res]
        # reverting bpe
        if revert_bpe:
            mark = False
            newelem = []
            for word in greedy_res_str:
                if mark:
                    newelem[-1] = newelem[-1][:-2] + word  # remove @@ from the last element and merge
                    mark = False
                else:
                    newelem.append(word)
                if word[-2:] == "@@":
                    mark = True
            greedy_res_str = newelem
        return greedy_res_str[1:-1]


class BeamSearch:
    def __init__(self, source_path: str, target_path: str, bpe: BPE, voc: Dictionary):
        self.ratio = calculate_ratio(source_path, target_path)
        self.bpe = bpe
        self.voc = voc
        # function for reverse Dictio access
        self.reverse_voc = {v: k for k, v in voc.getDictionary().items()}

    def beam_search(self, model: nn.Module, beam_size: int, dataloader: DataLoader):
        mapping = self.voc.getDictionary()
        results = [([mapping["<s>"]], 0.0)]
        early_s = []
        for batch in dataloader:  # batch is only 1 line
            x, _, blank = batch
            temp = []
            for elem, prob in results:
                last_ws = elem[-model.windowsize:]  # last window size elements
                if len(last_ws) < model.windowsize:  # padding
                    last_ws = [mapping["<s>"]] * (model.windowsize - len(last_ws)) + last_ws
                model.eval()
                out = model((x, torch.tensor(last_ws).unsqueeze(0), blank))
                out = F.softmax(out, dim=1)  # out is a column vec (1,voc_size)
                # get the best beam_size elements
                (values, indices) = torch.topk(out, beam_size)
                values, indices = values.squeeze(), indices.squeeze()
                for val, ind in zip(values, indices):  # iterate over beam_size
                    if ind.item() == mapping["</s>"]:
                        # </s> handling
                        early_s.append((elem + [ind.item()], prob + np.log(val.item())))
                    else:
                        temp.append((elem + [ind.item()], prob + np.log(val.item())))

                # convert back to string for debugging
                #string_sequences = [([self.reverse_voc.get(token, '<unk>') for token in sequence], probab) for sequence, probab in temp]

            results = sorted(temp, key=lambda z: z[1], reverse=True)[:beam_size]  # get the best beam_size elements
        return results + early_s

    def beam_wrapper(self, model: nn.Module, source_sentence: list[str], beam_size: int, offset: int, fill: int,
                     k_best: bool = False, revert_bpe: bool = True):
        filtered_source = self.voc.filterInput(self.bpe.applyBPE([source_sentence]))
        filtered_source[0] += ["</s>"] * fill
        pred_len = round(len(filtered_source[0]) * self.ratio)  # filtered_source has only one element
        results = []

        for length in range(pred_len - offset, pred_len + offset + 1):
            padding = [["<s>"] * length]
            batch = Batch(filtered_source, padding, length + 1, model.windowsize, self.voc, self.voc)
            data = TranslationDataset(batch.batches)
            dataloader = DataLoader(data, batch_size=1, shuffle=False)
            res = self.beam_search(model, beam_size, dataloader)
            # normalize prob's with length and convert back to probabilities
            res = [(elem, np.exp(prob / len(elem))) for (elem, prob) in res]
            # concat
            results += res

        beam_size_results = sorted(results, key=lambda x: x[1], reverse=True)[:beam_size]
        # convert back to string
        beam_size_results = [([self.reverse_voc.get(token, '<unk>') for token in sequence], probab) for sequence, probab
                             in beam_size_results]
        # reverting bpe
        if revert_bpe:
            for i, (elem, prob) in enumerate(beam_size_results):
                mark = False
                newelem = []
                for word in elem:
                    if mark:
                        newelem[-1] = newelem[-1][:-2] + word  # remove @@ from the last element and merge
                        mark = False
                    else:
                        newelem.append(word)
                    if word[-2:] == "@@":
                        mark = True
                beam_size_results[i] = (newelem, prob)
        # print k_best
        if k_best:
            print(f"top {beam_size} results:")
            for sequence, prob in beam_size_results:
                seq = sequence[1:-1]
                for token in seq:
                    print(token, end=" ")
                print(prob)
        # top element
        top_result, prob = beam_size_results[0]
        return top_result[1:-1]  # removing <s>, </s>

    def calc_bleus(self, model_path: str, source_eval: list[list[str]], target_eval: list[list[str]], beam_size: int,
                   offset: int, fill: int):
        # listing all modelnames in the directory
        path = Path(model_path)
        all_model_names = [file.name for file in path.rglob('*') if file.is_file() and not file.name.startswith(".")]
        print("Models loaded")
        # list of bleu_scores archieved with the different models
        bleu_list = []
        for model_name in all_model_names:
            model = Loader.load_model(model_name)
            print(f"Model: {model_name}")
            hyp = []
            # translating setences one by one
            for i, sequence in enumerate(source_eval):
                hyp.append(self.beam_wrapper(model, sequence, beam_size, offset, fill))
                # progress
                if i % 50 == 0:
                    print(f"Progress: {i}/{len(source_eval)}")
            korp = Korpus(target_eval, hyp)
            print(f"BLEU: {korp.bleu4()}")
            bleu_list.append((korp.bleu4(), model_name))
        return bleu_list


def calculate_ratio(source_path: str, target_path: str):
    source = fh1.read_file_words(source_path)
    target = fh1.read_file_words(target_path)

    source_avg = sum([len(s) for s in source]) / len(source)
    target_avg = sum([len(t) for t in target]) / len(target)

    return target_avg / source_avg


def calc_bleus(search: str, model_path: str, source_path, target_path, bpe, voc, beam_size: int, offset: int,
               fill: int):
    # listing all modelnames in the directory
    source_eval = fh1.read_file_words(args.eval_de)
    target_eval = fh1.read_file_words(args.eval_en)
    path = Path(model_path)
    all_model_names = [file.name for file in path.rglob('*') if file.is_file() and not file.name.startswith(".")]
    print("Models loaded")
    # list of bleu_scores archieved with the different models
    bleu_list = []
    for model_name in all_model_names:
        model = Loader.load_model(model_name)
        print(f"Model: {model_name}")
        hyp = []
        # translating setences one by one
        if search == "beam":
            bs = BeamSearch(source_path, target_path, bpe, voc)
            for i, sequence in enumerate(source_eval):
                hyp.append(bs.beam_wrapper(model, sequence, beam_size, offset, fill))
                # progress
                if i % 50 == 0:
                    print(f"Progress: {i}/{len(source_eval)}")
        elif search == "greedy":
            gs = GreedySearch(source_path, target_path, bpe, voc)
            for i, sequence in enumerate(source_eval):
                hyp.append(gs.greedy_wrapper(model, sequence, 0, True))
                # progress
                if i % 50 == 0:
                    print(f"Progress: {i}/{len(source_eval)}")
        else:
            print("wrong search method")
            return
        korp = Korpus(target_eval, hyp)
        print(f"BLEU: {korp.bleu4()}")
        bleu_list.append((korp.bleu4(), model_name))
    return bleu_list


def save_translations(self, model_path: str, source_eval: list[list[str]], target_eval: list[list[str]],
                      beam_size: int, offset: int, fill: int):
    path = Path(model_path)
    all_model_names = [file.name for file in path.rglob('*') if file.is_file() and not file.name.startswith(".")]
    for model_name in all_model_names:
        model = Loader.load_model(model_name)
        print(f"Model: {model_name}")
        hyp = []
        # translating setences one by one
        for i, sequence in enumerate(source_eval):
            hyp.append(self.beam_wrapper(model, sequence, beam_size, offset, fill))
            # progress
            if i % 50 == 0:
                print(f"Progress: {i}/{len(source_eval)}")
        with open(f"../config/translations/{model_name}.txt", "w") as f:
            for sentence in hyp:
                f.write(" ".join(sentence) + "\n")


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
    #model = Loader.load_model(args.model)
    """
    source = fh1.read_file_words(args.eval_de)
    gs = GreedySearch(args.train_de, args.train_en, bpe, voc)
    s1 = ["drei", "alte", "männer", "dfpiadofnoa", "einem", "anderen", "mann", "zu", ",", "wie", "er", "fisch", "zubereitet",
          "."]
    gs.greedy_wrapper(model, s1, 0, True)"""

    bs = BeamSearch(args.train_de, args.train_en, bpe, voc)
    source = fh1.read_file_words(args.eval_de)
    target = fh1.read_file_words(args.eval_en)
    #s1 = source[20]
    #bs.beam_wrapper(model, s1, args.beam_size, args.offset, True, True)

    print(calc_bleus("beam", "../config/Models", args.eval_de, args.eval_en, bpe, voc, 2, 1, 8))
    #print(bs.calc_bleus("../config/Models", source, target, 2, 1, 8))
