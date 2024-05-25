import math

import numpy

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

    def saveBatches(batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]], path: str, max_digits: int = 5):
        with open(path, "wb") as f:
            for (s, t, l) in batches:
                for i in range(len(s)):
                    # iterate over each row and combine them left-aligned with a tab in between
                    combinedrow = ' '.join(
                        '{{:<{}}}'.format(max_digits).format(item) for item in s[i]) + '\t' + '|' + '\t' + ' '.join(
                        '{{:<{}}}'.format(max_digits).format(item) for item in t[i]) + '\t' + '|' + '\t' + ' '.join(
                        '{{:<{}}}'.format(max_digits).format(item) for item in l[i]) + '\n'
                    f.write(combinedrow.encode())
                f.write("\n\n\n".encode())


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

    def substitutePair(self, word: str, pair: str) -> str:
        # it's possible, that we have to do the same merge multiple times in the same word
        # newWord is the word after each word
        newWord = word
        index = word.find(pair)
        # we have to ensure that we only merge separate tokens
        while index != -1:
            # check if first part of pair is a separate token
            if index == 0 or newWord[index - 1] == " ":
                # check if second part of pair is a separate token (either end of word or whitespace)
                if index + len(pair) == len(newWord) or newWord[index + len(pair)] == " ":
                    newWord = newWord[:index] + pair.replace(" ", "") + newWord[index + len(pair):]
                    index = newWord.find(pair, index + len(pair) - 1)  # -1 because we replaced the string
                else:
                    index = newWord.find(pair, index + 1)
            else:
                index = newWord.find(pair, index + 1)
        return newWord

    def mergeTokens(self, pairdict: dict[str, str]) -> dict[str, int]:
        # dictionary might be empty, if we have no pairs left (i.e. all tokens are merged)
        if not pairdict:
            return self.countTokenPairs()
        # find most frequent pair
        maxpair = max(pairdict, key=lambda k: (pairdict[k], k))
        # print(f"merge: {maxpair}")
        self.merges.append(tuple(maxpair.split()))
        # update worddict after finding most frequent pair
        newVocab = {}
        for word in self.worddict:
            # replace most frequent pair with merged pair but (!) differentiate between </w> and no </w>
            newWord = self.substitutePair(word, maxpair)
            newVocab[newWord] = self.worddict[word]

        self.worddict = newVocab
        # update pairdict
        return self.countTokenPairs()

    def learnBPE(self, iterations: int):
        pairdict = self.countTokenPairs()
        for i in range(iterations):
            print(f"Progress: {i}/{iterations}")
            pairdict = self.mergeTokens(pairdict)
        # print(f"Finished learning BPE with {iterations} iterations")

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
                """
                newWord = re.sub(r'^' + re.escape(" ".join(merge)) + r'(?=\s|$)', "".join(merge),
                                 appliedworddict[word])  # for start of the string
                newWord = re.sub(r'(?<=\s)' + re.escape(" ".join(merge)) + r'(?=\s|$)', "".join(merge),
                                 newWord)  # for surrounded by whitespaces

                """
                # without regex
                newWord = self.substitutePair(appliedworddict[word], " ".join(merge))
                appliedworddict[word] = newWord
        # use @@ as "in word" separator instead of </w>
        for word in appliedworddict:
            # replace whitespace with @@
            appliedworddict[word] = appliedworddict[word].replace(" ", "@@")
            # remove end of word token </w>
            appliedworddict[word] = appliedworddict[word].replace("</w>", "")

        # iterate over text and replace words with tokenized words
        text = [sublist[:] for sublist in data]
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
        counter = 4
        dictdict["<s>"] = 0
        dictdict["</s>"] = 1
        dictdict["<unk>"] = 2
        dictdict["<unk>@@"] = 3
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
                # add last element without @@
                newlist[i].append(l[(len(l) - 1)])
        return newlist

    def filterInput(self, text: list[list[str]]) -> list[list[str]]:
        splittedtext = self.splitintosubwords(text)
        # filter out words that are not in the dictionary
        for i in range(len(splittedtext)):
            for j in range(len(splittedtext[i])):
                if splittedtext[i][j] not in self.dict:
                    ## nachträglich hinzugefügt:
                    if splittedtext[i][j][-2:] == "@@":
                        splittedtext[i][j] = "<unk>@@"
                    else:
                        splittedtext[i][j] = "<unk>"
        return splittedtext


class Batch:

    def __init__(self,source: list[list[str]], target: list[list[str]], batchsize: int, windowsize: int,
                      voc_source: Dictionary, voc_target: Dictionary):
        self.batches = Batch.createBatches(source, target, batchsize, windowsize,
                      voc_source, voc_target)
        self.windowsize = windowsize # is needed as parameter for the model
        self.voc_len = len(voc_source.dict) # dict will be a combined dict in our case so voc_source = vod_target
    @staticmethod
    def createSmallBatch(source: list[str],
                         target: list[str],
                         sequencelen: int,
                         windowsize: int,
                         voc_source: Dictionary,
                         voc_target: Dictionary) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """create a batch of one sentence pair"""
        vocmapping_source = voc_source.getDictionary()
        vocmapping_target = voc_target.getDictionary()
        S = np.zeros((sequencelen, windowsize * 2 + 1), dtype=int)
        T = np.zeros((sequencelen, windowsize), dtype=int)
        L = np.zeros((sequencelen, 1), dtype=int)

        alignmentpar = len(source) / len(target)

        for i in range(sequencelen):
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

    @staticmethod
    def fillUpwithPadding(batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]], tofillup: int,
                          windowsize: int,
                          voc_source: Dictionary, voc_target: Dictionary):
        """fill up the last batch with padding to reach the desired batchsize"""
        s_pad = np.full((tofillup, windowsize * 2 + 1), voc_source.getDictionary()["</s>"])
        t_pad = np.full((tofillup, windowsize), voc_target.getDictionary()["</s>"])
        l_pad = np.full((tofillup, 1), voc_target.getDictionary()["</s>"])
        (s, t, l) = batches[-1]
        s1 = np.concatenate((s, s_pad))
        t1 = np.concatenate((t, t_pad))
        l1 = np.concatenate((l, l_pad))
        batches[-1] = (s1, t1, l1)

    @staticmethod
    def createBatches(source: list[list[str]], target: list[list[str]], batchsize: int, windowsize: int,
                      voc_source: Dictionary, voc_target: Dictionary) \
            -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """ for each sentence pair in the input, create a batch
             after that we want to merge those small batches into bigger ones"""

        # initialize the batches array i.e. appending first (empty) element
        batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        batches.append((numpy.empty([0, 2 * windowsize + 1], dtype=int), numpy.empty([0, windowsize], dtype=int),
                        numpy.empty([0, 1], dtype=int)))
        while len(target) > 0:
            (s_new, t_new, l_new) = Batch.createSmallBatch(source[0], target[0], len(target[0]) + 1, windowsize,
                                                           voc_source, voc_target)

            (s, t, l) = batches[-1] # get last batch i.e. the batch we want to append to
            if len(target[0]) + 1 + len(s) > batchsize:  # small batch too long
                # splitting small batch -> filling up the last batch and creating a new one
                s1 = np.concatenate((s, s_new[:(batchsize - len(s))]))
                t1 = np.concatenate((t, t_new[:(batchsize - len(s))]))
                l1 = np.concatenate((l, l_new[:(batchsize - len(s))]))
                batches[-1] = (s1, t1, l1)
                for i in range(math.ceil(len(s)/batchsize)):
                    batches.append(
                        (numpy.empty([0, 2 * windowsize + 1], dtype=int), numpy.empty([0, windowsize], dtype=int),
                         numpy.empty([0, 1], dtype=int)))
                    (s, t, l) = batches[-1]
                    s1 = np.concatenate((s, s_new[(batchsize - len(s))+i*batchsize:(batchsize - len(s))+(i+1)*batchsize]))
                    t1 = np.concatenate((t, t_new[(batchsize - len(s))+i*batchsize:(batchsize - len(s))+(i+1)*batchsize]))
                    l1 = np.concatenate((l, l_new[(batchsize - len(s))+i*batchsize:(batchsize - len(s))+(i+1)*batchsize]))
                    batches[-1] = (s1, t1, l1)
            else:  # concatenating small batch to batch
                s1 = np.concatenate((s, s_new))
                t1 = np.concatenate((t, t_new))
                l1 = np.concatenate((l, l_new))
                batches[-1] = (s1, t1, l1)
            target = target[1:]
            source = source[1:]
        # fill up the last batch with padding
        #Batch.fillUpwithPadding(batches, batchsize - len(batches[-1][0]), windowsize, voc_source, voc_target)
        return batches

    @staticmethod
    def matrElemToString(matr: np.ndarray, vocab: Dictionary) -> np.ndarray:
        mapping = vocab.getDictionary()
        newmatr = np.zeros((matr.shape[0], matr.shape[1]), dtype=np.dtype('U100'))
        # for each elem, which is a number, change it to the corresponding word
        for i in range(len(matr)):
            for j in range(len(matr[i])):
                print(f"Progress: {i}/{len(matr)}")
                test = list(mapping.keys())[list(mapping.values()).index(matr[i][j])]  # super slow
                newmatr[i][j] = test
        return newmatr

    @staticmethod
    def getBatchesWithStrings(batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                              vocab_source: Dictionary,
                              vocab_target: Dictionary) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        newbatches = []
        for i in range(len(batches)):
            # progress bar
            print(f"Progress: {i}/{len(batches)}")
            (s, t, l) = batches[i]
            s = Batch.matrElemToString(s, vocab_source)
            t = Batch.matrElemToString(t, vocab_target)
            l = Batch.matrElemToString(l, vocab_target)
            newbatches.append((s, t, l))
        return newbatches


class Analysis:
    @staticmethod
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
            results.append((len(dictio_source.getDictionary()) - 3, len(dictio_target.getDictionary()) - 3,
                            len(dictio_combined.getDictionary()) - 3))

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
        bars1 = plt.bar(r1, source_results, color='b', width=bar_width, edgecolor='grey', label='Source')
        bars2 = plt.bar(r2, target_results, color='g', width=bar_width, edgecolor='grey', label='Target')
        bars3 = plt.bar(r3, combined_results, color='r', width=bar_width, edgecolor='grey', label='Combined')

        Analysis.add_value_labels(bars1)
        Analysis.add_value_labels(bars2)
        Analysis.add_value_labels(bars3)

        # Add xticks on the middle of the group bars
        plt.xlabel('Number of BPE operations', fontweight='bold')
        plt.xticks([r + bar_width for r in range(len(source_results))], bpe_operations)

        # Create legend & Show graphic
        plt.legend()
        plt.show()

    @staticmethod
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height,
                     '{0:.0f}'.format(height), ha='center', va='bottom')

    @staticmethod
    def analyzeBPEWrapper():
        parser = argparse.ArgumentParser()
        parser.add_argument("--source", type=str, help="Path to source file")
        parser.add_argument("--target", type=str, help="Path to target file")
        args = parser.parse_args()
        Analysis.analyzeBPE(args.source, args.target)

    @staticmethod
    def printBatches():
        parser = argparse.ArgumentParser()

        parser.add_argument("--englisch", type=str, help="Path to training file")
        parser.add_argument("--deutsch", type=str, help="Path to input file")
        parser.add_argument("--output_int", type=str, help="Path to output file")
        parser.add_argument("--output_str", type=str, help="Path to output file")
        args = parser.parse_args()

        source_input = fh.read_file_words(args.deutsch)
        target_input = fh.read_file_words(args.englisch)
        # lines 1100 to 1200 of both inputs
        combined_input = source_input[1100:1200] + target_input[1100:1200]
        bpe_mixed = BPE(FileHandler.flattenList(combined_input))
        bpe_mixed.learnBPE(1000)
        dictio_mixed = Dictionary(bpe_mixed.applyBPE(combined_input))

        batches = Batch.createBatches(dictio_mixed.filterInput(bpe_mixed.applyBPE(source_input)), dictio_mixed.filterInput(bpe_mixed.applyBPE(target_input)), 200, 5, dictio_mixed,
                                      dictio_mixed)
        str_batches = Batch.getBatchesWithStrings(batches, dictio_mixed, dictio_mixed)
        FileHandler.saveBatches(batches, args.output_int, 4)
        FileHandler.saveBatches(str_batches, args.output_str, 20)



def main():
    Analysis.printBatches()

if __name__ == "__main__":
    main()
