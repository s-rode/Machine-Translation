from Blatt1 import FileHandler as fh

import argparse


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

    def mergeTokens(self, pairdict: dict[str,str]) -> dict[str, int]:
        # find most frequent pair
        maxpair = max(pairdict, key=pairdict.get)
        self.merges.append(tuple(maxpair.split())) # todo check if tuple is correct
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to input file")
    args = parser.parse_args()
    bpe = BPE(args.input)
    bpe.learnBPE(1000)
    list = bpe.getMerges()
    print(list)
    print(bpe.revertBPEEnconding(bpe.applyBPE(args.input)))

if __name__ == "__main__":
    main()