from Blatt1 import FileHandler as fh

import argparse
import numpy as np
import matplotlib.pyplot as plt


class FileHandler:
    @staticmethod
    def createFlatInput(data) -> list[str]:
        words: list[list[str]] = fh.read_file_words(data)
        return [item for sublist in words for item in sublist]

    def flattenList(data: list[list[str]]) -> list[str]:
        return [item for sublist in data for item in sublist]


class BPE:
    worddict: dict[str, int]
    # save merges as pairs
    merges: list[tuple[str, str]]

    def __init__(self, inputlist: list[str]):
        self.merges = []
        self.worddict = self.buildDictionary(inputlist)
        self.worddict = self.splitDictionary(self.worddict)

    def getDict(self):
        return self.worddict

    def buildDictionary(self, text: list[str]) -> dict[str, int]:
        dict = {}
        for word in text:
            if word in dict:
                dict[word] += 1
            else:
                dict[word] = 1
        return dict

    def splitDictionary(self, lexicon: dict[str, int]) -> dict[str, int]:
        # split dictionary entries into tokens seperated by whitespace and add end of line token
        newDict = {}
        for key in lexicon:
            tokenizedWord = " ".join(list(key))
            tokenizedWord += "</w>"
            newDict[tokenizedWord] = lexicon[key]
        return newDict

    def countTokenPairs(self) -> dict[str, int]:
        pairdict = {}
        for word in self.worddict:
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = symbols[i] + " " + symbols[i + 1]
                if pair in pairdict:
                    pairdict[pair] += self.worddict[word]
                else:
                    pairdict[pair] = self.worddict[word]
        return pairdict

    def mergeTokens(self, pairdict: dict[str, str]) -> dict[str, int]:
        # dictionary might be empty, if we have no pairs left (i.e. all tokens are merged)
        if not pairdict:
            return self.countTokenPairs()

        # find most frequent pair
        maxpair = max(pairdict, key=pairdict.get)
        self.merges.append(tuple(maxpair.split()))  # todo check if tuple is correct
        newVocab = {}
        for word in self.worddict:
            newWord = word.replace(maxpair, maxpair.replace(" ", ""))
            newVocab[newWord] = self.worddict[word]

        self.worddict = newVocab
        # update pairdict
        return self.countTokenPairs()

    def learnBPE(self, iterations: int):
        pairdict = self.countTokenPairs()
        for i in range(iterations):
            # fortschrittbalken
            print(f"Progress: {i}/{iterations}")
            pairdict = self.mergeTokens(pairdict)

    def getMerges(self) -> list[tuple[str, str]]:
        return self.merges

    def buildAppliedWordDict(self, inputStream: list[str]) -> dict[str, str]:
        # iterate over words in input and create dictionary with word as key and splitted word as value
        appliedworddict = {}
        for word in inputStream:
            if word not in appliedworddict:
                appliedworddict[word] = " ".join(list(word)) + "</w>"
        return appliedworddict

    def applyBPE(self, data: list[list[str]]) -> list[list[str]]:
        appliedworddict = self.buildAppliedWordDict(FileHandler.flattenList(data))
        # apply merges to dictionary
        for merge in self.merges:
            for word in appliedworddict:
                # replace word containing tokens to merge with word with merged tokens
                appliedworddict[word] = appliedworddict[word].replace(" ".join(merge), "".join(merge))

        # use @@ as "in word" separator instead of </w>
        for word in appliedworddict:
            # replace whitespace with @@
            appliedworddict[word] = appliedworddict[word].replace(" ", "@@")
            # remove end of word token </w>
            appliedworddict[word] = appliedworddict[word].replace("</w>", "")

        # iterate over text and replace words with tokenized words
        text = [sublist.copy() for sublist in data]
        for i in range(len(text)):
            for j in range(len(text[i])):
                text[i][j] = appliedworddict[text[i][j]]
        return text

    def revertBPEEnconding(self, data: list[list[str]]) -> list[list[str]]:
        # revert tokenization
        for i in range(len(data)):
            for j in range(len(data[i])):
                # remove @@
                data[i][j] = data[i][j].replace("@@", "")
        return data


class Dictionary:
    def __init__(self, text: list[list[str]]):
        self.dict = {}
        self.dict = self.buildDictdict(text)

    def add(self, key, value):
        self.dict[key] = value

    def get(self, key):
        return self.dict[key]

    def contains(self, key):
        return key in self.dict

    def getDictionary(self):
        return self.dict

    def buildDictdict(self, text: list[list[str]]) -> dict[str, int]:
        # bidirectional map between Strings and indices
        dictdict = {}
        counter = 2
        dictdict["<s>"] = 0
        dictdict["</s>"] = 1
        inputList = self.splitintosubwords(text)
        for i in range(len(inputList)):
            for j in range(len(inputList[i])):
                if inputList[i][j] not in dictdict:
                    dictdict[inputList[i][j]] = counter
                    counter += 1
        return dictdict

    def splitintosubwords(self, text: list[list[str]]) -> list[list[str]]:
        # split after each @@ while keeping the @@
        newlist: list[list[str]] = []
        for i in range(len(text)):
            newlist.append([])
            for j in range(len(text[i])):
                l = text[i][j].split("@@")
                for k in range(len(l) - 1):
                    newlist[i].append(l[k] + "@@")
                newlist[i].append(l[(len(l) - 1)])
        return newlist

    def filterInput(self, text: list[list[str]]) -> list[list[str]]:
        splittedtext = self.splitintosubwords(text)
        # filter out words that are not in the dictionary
        for i in range(len(splittedtext)):
            for j in range(len(splittedtext[i])):
                if splittedtext[i][j] not in self.dict:
                    splittedtext[i][j] = "<unk>"
        return splittedtext


def createSmallBatch(source: list[str],
                     target: list[str],
                     batchsize: int,
                     windowsize: int,
                     voc_source: Dictionary,
                     voc_target: Dictionary) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vocmapping_source = voc_source.getDictionary()
    vocmapping_target = voc_target.getDictionary()
    S = np.zeros((batchsize, windowsize * 2 + 1), dtype=int)
    T = np.zeros((batchsize, windowsize), dtype=int)
    L = np.zeros((batchsize, 1), dtype=int)

    alignmentpar = len(source) / len(target)

    for i in range(batchsize):
        b_i = round(alignmentpar * i)
        for j in range(2 * windowsize + 1):
            if b_i - windowsize + j < 0:
                S[i, j] = vocmapping_source["<s>"]
            elif b_i - windowsize + j >= len(source):
                S[i, j] = vocmapping_source["</s>"]
            else:
                S[i, j] = vocmapping_source[source[b_i - windowsize + j]]
        for j in range(windowsize):
            if i - windowsize + j < 0:
                T[i, j] = vocmapping_target["<s>"]
            elif i - windowsize + j >= len(target):
                T[i, j] = vocmapping_target["</s>"]
            else:
                T[i, j] = vocmapping_target[target[i - windowsize + j]]
        if i < len(target):
            L[i] = vocmapping_target[target[i]]
        else:
            L[i] = vocmapping_target["</s>"]
    return S, T, L


def createBatches(source: list[list[str]], target: list[list[str]], batchsize: int, windowsize: int,
                  voc_source: Dictionary, voc_target: Dictionary) \
        -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """ for each sentence pair in the input, create a batch
         after that we want to merge those small batches into bigger ones"""

    batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    batchcounter = 0

    while len(target) > 0:
        if len(target[0]) > batchsize:
            # split list into two pieces if batchsize is less than list length
            # when this happens, listcount is 0
            target1 = [target[0][c] for c in range(int(len(target[0]) / 2))]
            target2 = [target[0][c] for c in range(int(len(target[0]) / 2), len(target[0]))]
            source1 = [source[0][c] for c in range(int(len(source[0]) / 2))]
            source2 = [source[0][c] for c in range(int(len(source[0]) / 2), len(source[0]))]
            target = target1 + target2 + target[1:]
            source = source1 + source2 + source[1:]
        elif batchcounter + len(target[0]) + 1 > batchsize:
            #todo: fill up the rest of the batch with padding
            batchcounter = 0
        else:
            (s_new, t_new, l_new) = createSmallBatch(source[0], target[0], len(target[0]) + 1, windowsize, voc_source,
                                                     voc_target)
            # if it doesn't exist yet, create a new batch
            if batchcounter == 0:
                batches.append((s_new, t_new, l_new))
            else:
                (s, t, l) = batches[-1]
                s1 = np.concatenate((s, s_new))
                t1 = np.concatenate((t, t_new))
                l1 = np.concatenate((l, l_new))
                batches[-1] = (s1, t1, l1)
            batchcounter += len(target[0]) + 1
            # pop the first element of the lists
            target = target[1:]
            source = source[1:]
    return batches


def matrElemToString(matr: np.ndarray, vocab: Dictionary) -> np.ndarray:
    mapping = vocab.getDictionary()
    newmatr = np.zeros((matr.shape[0], matr.shape[1]), dtype=np.dtype('U100'))
    # for each elem, which is a number, change it to the corresponding word
    for i in range(len(matr)):
        for j in range(len(matr[i])):
            test = list(mapping.keys())[list(mapping.values()).index(matr[i][j])]
            newmatr[i][j] = test
    return newmatr


def getBatchesWithStrings(batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                          vocab_source: Dictionary,
                          vocab_target: Dictionary) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    newbatches = []
    for i in range(len(batches)):
        # progress bar
        print(f"Progress: {i}/{len(batches)}")
        (s, t, l) = batches[i]
        s = matrElemToString(s, vocab_source)
        t = matrElemToString(t, vocab_target)
        l = matrElemToString(l, vocab_target)
        newbatches.append((s, t, l))
    return newbatches


def analyzeBPE(source: str, target: str):
    # input List[List]]
    source_input = fh.read_file_words(source)
    target_input = fh.read_file_words(target)
    combined_input = source_input + target_input

    # input List[str]
    source_flat = FileHandler.flattenList(source_input)
    target_flat = FileHandler.flattenList(target_input)
    combined_flat = FileHandler.flattenList(combined_input)

    # save results in a list to plot later
    results = []

    for i in [1000, 5000, 15000]:
        # training on source language
        bpe_source = BPE(source_flat)
        bpe_source.learnBPE(i)
        dictio_source = Dictionary(bpe_source.applyBPE(source_input))
        # training on target language
        bpe_target = BPE(target_flat)
        bpe_target.learnBPE(i)
        dictio_target = Dictionary(bpe_target.applyBPE(target_input))
        # training on combined language
        bpe_combined = BPE(combined_flat)
        bpe_combined.learnBPE(i)
        dictio_combined = Dictionary(bpe_combined.applyBPE(combined_input))
        # save results
        results.append((len(dictio_source.getDictionary())-2, len(dictio_target.getDictionary())-2, len(dictio_combined.getDictionary())-2))


    # Define the number of BPE operations
    bpe_operations = [1000, 5000, 15000]

    # Define the results
    source_results = [results[0][0], results[1][0], results[2][0]]
    target_results = [results[0][1], results[1][1], results[2][1]]
    combined_results = [results[0][2], results[1][2], results[2][2]]

    # Define the bar width
    bar_width = 0.25

    # Define the position of the bars on the x-axis
    r1 = np.arange(len(source_results))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Create the bars
    plt.bar(r1, source_results, color='b', width=bar_width, edgecolor='grey', label='Source')
    plt.bar(r2, target_results, color='g', width=bar_width, edgecolor='grey', label='Target')
    plt.bar(r3, combined_results, color='r', width=bar_width, edgecolor='grey', label='Combined')

    # Add xticks on the middle of the group bars
    plt.xlabel('Number of BPE operations', fontweight='bold')
    plt.xticks([r + bar_width for r in range(len(source_results))], bpe_operations)

    # Create legend & Show graphic
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    """
    parser.add_argument("--training", type=str, help="Path to training file")
    parser.add_argument("--input", type=str, help="Path to input file")
    args = parser.parse_args()
    bpe = BPE(args.input)
    bpe.learnBPE(1000)
    dictio = Dictionary(bpe.applyBPE(args.training))"""

    """test batches:

    parser.add_argument("--englisch", type=str, help="Path to training file")
    parser.add_argument("--deutsch", type=str, help="Path to input file")
    args = parser.parse_args()
    bpe_de = BPE(FileHandler.createFlatInput(args.deutsch))
    bpe_de.learnBPE(1000)
    print(bpe_de.getDict())
    dictio_de = Dictionary(bpe_de.applyBPE(args.deutsch))
    print(dictio_de.getDictionary())
    bpe_en = BPE(FileHandler.createFlatInput(args.englisch))

    bpe_en.learnBPE(2000)
    dictio_en = Dictionary(bpe_en.applyBPE(args.englisch))
    # create batches
    batches = createBatches(dictio_en.filterInput(bpe_en.applyBPE(args.englisch)),
                            dictio_de.filterInput(bpe_de.applyBPE(args.deutsch)),
                            200, 5, dictio_en, dictio_de)
    print(getBatchesWithStrings(batches, dictio_en, dictio_de))
    """

    # analyze BPE
    parser.add_argument("--source", type=str, help="Path to source file")
    parser.add_argument("--target", type=str, help="Path to target file")
    args = parser.parse_args()
    analyzeBPE(args.source, args.target)


if __name__ == "__main__":
    # print("@@".join("Hallo@@test".split("@@")))
    main()
