import torch
import torch.nn as nn
import torch.nn.functional as F
import Model

from Blatt2 import BPE, Dictionary, Batch, FileHandler
from Model import TranslationDataset
from torch.utils.data import DataLoader, Dataset


def score(model: nn.Module, source_sentences, target_sentences, bpe: BPE, voc: Dictionary):
    windowsize = int(model.fc_target.weights.shape[0]/model.embedding.weight.shape[1])
    model.eval()
    with torch.no_grad():
        scores = []
        for source, target in zip(source_sentences, target_sentences):
            dataloader = create_sentence_dataloader(source, target, bpe, voc, windowsize)
            batch = next(iter(dataloader))
            _, _, y = batch
            y_w = model(batch)
            score = calculate_score(y_w, y)
            scores.append(score)
        return scores


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

