import numpy as np
import argparse


class FileHandler:
    @staticmethod
    def read_file_words(file) -> list[list[str]]:
        with open(file, 'r', encoding="utf-8") as f:
            sentences = [line.split() for line in f]
            f.close()
        return sentences


class Korpus:
    data = []
    hyplen = 0
    reflen = 0

    def __init__(self, ref_file, hyp_file):
        self.data = list(zip(FileHandler.read_file_words(ref_file), FileHandler.read_file_words(hyp_file)))
        for (r, h) in self.data:
            self.hyplen += len(h)
            self.reflen += len(r)

    @staticmethod
    def levenshtein(a, b, shouldprint: bool):
        lenA = len(a) + 1
        lenB = len(b) + 1
        if lenA == 0 or lenB == 0:
            print("either a or b are empty")
            return max(lenA, lenB) - 1
        # init as zeros
        matr = np.zeros([lenA, lenB], dtype=int)

        # fill first column
        for i in range(lenA):
            matr[i][0] = i

        # fill first row
        for j in range(lenB):
            matr[0][j] = j

        for i in range(1, lenA):
            for j in range(1, lenB):
                if a[i - 1] == b[j - 1]:
                    matr[i][j] = matr[i - 1][j - 1]
                else:
                    matr[i][j] = min(matr[i - 1][j - 1], matr[i][j - 1], matr[i - 1][j]) + 1

        if shouldprint:
            i = lenA - 1
            j = lenB - 1
            sol = []
            while (i > 0) or (j > 0):
                if matr[i - 1][j] == matr[i][j] - 1:
                    sol.append(f"insert {a[i - 1]}")
                    i = i - 1
                elif matr[i][j - 1] == matr[i][j] - 1:
                    sol.append(f"delete {b[j - 1]}")
                    j = j - 1
                elif matr[i - 1][j - 1] == matr[i][j]:
                    i = i - 1
                    j = j - 1
                    sol.append("match")
                else:
                    sol.append(f"substitute {a[i - 1]} with {b[j - 1]}")
                    i = i - 1
                    j = j - 1
            solStr = "; ".join(sol)
            print("Lösungsweg: ", solStr)

        return matr[lenA - 1][lenB - 1]

    def per(self):
        matches = 0
        for (r, h) in self.data:
            matches += Korpus.ngram_matching(h, r, 1)
        if self.reflen == 0:
            raise Exception("Division by zero")
        return 1 - ((matches - max(0, self.hyplen - self.reflen)) / self.reflen)

    def wer(self):
        lvDist = 0
        for (r, h) in self.data:
            lvDist += Korpus.levenshtein(r, h, False)
        if self.reflen == 0:
            raise Exception("Division by zero")
        return lvDist / self.reflen

    def ngram_precision(self, n: int):
        nominator = 0
        denominator = 0
        for (r, h) in self.data:
            nominator += Korpus.ngram_matching(h, r, n)
            denominator += len(Korpus.fetch_ngrams(h, n))
        if denominator == 0:
            raise Exception("Division by zero")
        return nominator / denominator

    @staticmethod
    def ngram_matching(h, r, n):
        count = 0
        lHyp = Korpus.fetch_ngrams(h, n)
        lRef = Korpus.fetch_ngrams(r, n)
        for hyp in lHyp:
            for ref in lRef:
                if hyp == ref:
                    count += 1
                    lRef.remove(ref)
                    # if we found a match, we delete the reference n-gram from lRef to avoid multiple matches
                    # breaking the inner loop -> outer loop goes into next iteration and doesn't count the same
                    # n-gram again
                    break
        return count

    def brevity_penalty(self):
        c = self.hyplen
        r = self.reflen
        if c > r:
            return 1
        elif c:
            return np.exp(1 - (r / c))
        else:  # c == 0
            return 0

    @staticmethod
    def fetch_ngrams(seq, n):
        gramlist = []
        for i in range(0, len(seq) - n + 1):
            gramlist.append(seq[i:i + n])  # i is max. len(h) - n => we're slicing max. to len(h) - 1
        return gramlist

    def bleu4(self):
        bp = self.brevity_penalty()
        s = 0
        for i in range(1, 5):
            s += (1 / 4) * np.log(self.ngram_precision(i))
        return bp * np.exp(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref")
    parser.add_argument("--hyp")
    parser.add_argument("--metric")
    args = parser.parse_args()

    ref_lines = FileHandler.read_file_words(args.ref)
    hyp_lines = FileHandler.read_file_words(args.hyp)

    korpus = Korpus(args.ref, args.hyp)

    match args.metric:
        case "WER":
            print(korpus.wer())
        case "PER":
            print(korpus.per())
        case "BLEU":
            print(korpus.bleu4())


if __name__ == "__main__":
    main()
