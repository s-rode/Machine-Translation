import numpy as np
import argparse
from torcheval.metrics.functional import word_error_rate as wer
from torcheval.metrics.functional import bleu_score as bleu


class Metric:
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

    @staticmethod
    def per(data: list[tuple[[str], [str]]]):
        matches = 0
        hyplen = 0
        reflen = 0
        for (r, h) in data:
            matches += Metric.ngram_matching(h, r, 1)
            hyplen += len(h)
            reflen += len(r)
        if reflen == 0:
            raise Exception("Division by zero")
        return 1 - ((matches - max(0, hyplen - reflen)) / reflen)

    @staticmethod
    def wer(data: list[tuple[[str], [str]]]):
        refLength = 0
        lvDist = 0
        for (r, h) in data:
            lvDist += Metric.levenshtein(r, h, False)
            refLength += len(r)
        if refLength == 0:
            raise Exception("Division by zero")
        return lvDist / refLength

    @staticmethod
    def ngram_precision(data: list[tuple[[str], [str]]], n: int):
        nominator = 0
        denominator = 0
        for (r, h) in data:
            nominator += Metric.ngram_matching(h, r, n)
            denominator += len(Metric.fetch_ngrams(h, n))
        if denominator == 0:
            raise Exception("Division by zero")
        return nominator / denominator

    @staticmethod
    def ngram_matching(h, r, n):
        count = 0
        lHyp = Metric.fetch_ngrams(h, n)
        lRef = Metric.fetch_ngrams(r, n)
        for hyp in lHyp:
            for ref in lRef:
                if hyp == ref:
                    count += 1
                    lRef.remove(ref)
                    break
        return count

    @staticmethod
    def brevity_penalty(data: list[tuple[[str], [str]]]):
        c = r = 0
        for (ref, hyp) in data:
            c += len(hyp)
            r += len(ref)

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

    @staticmethod
    def bleu4(data: list[tuple[[str], [str]]]):
        bp = Metric.brevity_penalty(data)
        s = 0
        for i in range(1, 5):
            s += (1 / 4) * np.log(Metric.ngram_precision(data, i))
        return bp * np.exp(s)


class FileHandler:
    @staticmethod
    def read_file_words(file) -> list[list[str]]:
        with open(file, 'r', encoding="utf-8") as f:
            sentences = [line.split() for line in f]
            f.close()
        return sentences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref")
    parser.add_argument("--hyp")
    parser.add_argument("--metric")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    ref_lines = FileHandler.read_file_words(args.ref)
    hyp_lines = FileHandler.read_file_words(args.hyp)

    data = list(zip(ref_lines, hyp_lines))

    match args.metric:
        case "WER":
            print(Metric.wer(data))
            if args.compare:
                torchwer = wer([' '.join(sublist) for sublist in hyp_lines], [' '.join(sublist) for sublist in ref_lines])
                print("Torch WER: ", str(torchwer))
        case "PER":
            print(Metric.per(data))
        case "BLEU":
            print(Metric.bleu4(data))
            if args.compare:
                torchbleu = bleu([' '.join(sublist) for sublist in hyp_lines], [' '.join(sublist) for sublist in ref_lines])
                print("Torch BLEU: ", str(torchbleu))


if __name__ == "__main__":
    main()
