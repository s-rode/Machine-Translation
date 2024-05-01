from Blatt1 import FileHandler as fh

import argparse
import numpy as np


class BPE:
    worddict: dict[str, int]
    # save merges as pairs
    merges: list[tuple[str, str]]

    def __init__(self, input):
        self.merges = []
        self.worddict = self.buildDictionary(self.createInput(input))
        self.worddict = self.splitDictionary(self.worddict)

    def createInput(self, data) -> list[str]:
        words: list[list[str]] = fh.read_file_words(data)
        return [item for sublist in words for item in sublist]

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

    def applyBPE(self, data) -> list[list[str]]:
        appliedworddict = self.buildAppliedWordDict(self.createInput(data))
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
        text = fh.read_file_words(data)
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
        counter = 0
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


def createBatch(source: list[str],
                target: list[str],
                batchsize: int,
                windowsize: int,
                voc: Dictionary) -> tuple[np.matrix,np.matrix,np.matrix]:

    vocmapping = voc.getDictionary()
    S = np.zeros((batchsize, windowsize * 2 + 1))
    T = np.zeros((batchsize,windowsize))
    L = np.zeros((batchsize,1))

    alignmentpar = len(source)/len(target)

    for i in range(batchsize):
        b_i = alignmentpar * i
        for j in range(2*windowsize+1):
            if b_i-windowsize+j < 0:
                S[i,j] = vocmapping["<s>"]
            elif b_i-windowsize+j >= len(source):
                S[i,j] = vocmapping["</s>"]
            else:
                S[i,j] = vocmapping[source[b_i-windowsize+j]]
        for j in range(windowsize):
            if i-windowsize+j < 0:
                T[i,j] = vocmapping["<s>"]
            elif i-windowsize+j >= len(target):
                T[i,j] = vocmapping["</s>"]
            else:
                T[i,j] = vocmapping[target[i-windowsize+j]]
        L[i] = vocmapping[target[i]]

def createBatches(source: list[list[str]], target: list[list[str]], batchsize : int, windowsize : int, voc : Dictionary) -> list[tuple[np.matrix,np.matrix,np.matrix]]:
    # split full(!) sentences into lists such that each list contains max. of batchsize elements
    batches = []
    counter = 0
    oldcounter = 0
    listcount = 0

    #todo: if sentence is longer than batch
    while target:
        if len(target[listcount]) > batchsize:
            # split list into two pieces if batchsize is less than list length
            # when this happens, listcount is 0
            target1 = [elem in target[listcount] for elem in range(int(len(target[listcount])/2))]
            target2 = [elem in target[listcount] for elem in range(int(len(target[listcount])/2),len(target[listcount]))]
            source1 = [elem in source[listcount] for elem in range(int(len(source[listcount])/2))]
            source2 = [elem in source[listcount] for elem in range(int(len(source[listcount])/2),len(source[listcount]))]
            target = target[0:listcount] + target1 + target2 + target[listcount+1:]
            source = source[0:listcount] + source1 + source2 + source[listcount+1:]
        if counter-1 > batchsize:
            targetmerged = [elem for sublist in target[0:listcount] for elem in sublist]
            sourcemerged = [elem for sublist in source[0:listcount] for elem in sublist]
            batches.append(createBatch(sourcemerged,targetmerged,batchsize,windowsize,voc))
            target = target[listcount:]
            source = source[listcount:]
            listcount = 0
            counter = 0
        elif counter-1 <= batchsize:
            counter += len(target[listcount])
            listcount += 1




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=str, help="Path to training file")
    parser.add_argument("--input", type=str, help="Path to input file")
    args = parser.parse_args()
    bpe = BPE(args.input)
    bpe.learnBPE(100)
    list = bpe.getMerges()
    dictio = Dictionary(bpe.applyBPE(args.training))
    # print(list)
    print(dictio.filterInput(bpe.applyBPE(args.input)))


if __name__ == "__main__":
    # print("@@".join("Hallo@@test".split("@@")))
    main()
