import numpy as np
import argparse
from torcheval.metrics import BLEUScore, WordErrorRate
from torcheval.metrics.functional import word_error_rate as wer
from torcheval.metrics.functional import bleu_score as bleu
import torchaudio


def levenshtein(a, b, shouldprint: bool):
    lenA = len(a) + 1
    lenB = len(b) + 1
    if lenA == 0 or lenB == 0:
        print("either a or b are empty")
        return max(lenA,lenB) - 1
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


def levenhsteinWrapper(a, b, shouldprint: bool):
    return levenshtein(a.split(), b.split(), shouldprint)


def PER(data: list[tuple[str, str]]):
    matches = 0
    hyplen = 0
    reflen = 0
    for (r, h) in data:
        matches += nGramMatching(h, r, 1)
        hyplen += len(h.split())
        reflen += len(r.split())
    if reflen == 0:
        raise Exception("Division by zero")
    return 1 - ((matches - max(0, hyplen - reflen)) / reflen)


def WER(data: list[tuple[str, str]]):
    refLength = 0
    lvDist = 0
    lvDist2 = 0
    for (r, h) in data:
        lvDist += levenhsteinWrapper(r, h, False)
        lvDist2 += torchaudio.functional.edit_distance(r.split(),h.split())
        refLength += len(r.split())
    if refLength == 0:
        raise Exception("Division by zero")
    print("torch leven:"+str(lvDist2), "our leven:"+str(lvDist))
    print(refLength)
    return lvDist / refLength

def WER2(data: list[tuple[[str],[str]]]):
    refLength = 0
    lvDist = 0
    for (r, h) in data:
        lvDist += torchaudio.functional.edit_distance(r,h)
        refLength += len(r)
    if refLength == 0:
        raise Exception("Division by zero")
    print(refLength)
    return lvDist / refLength


def nGramPrecision(data: list[tuple[str, str]], n: int):
    nominator = 0
    denominator = 0
    for (r, h) in data:
        nominator += nGramMatching(h, r, n)
        denominator += len(fetchNGrams(h.split(), n))
    if (denominator == 0):
        raise Exception("Division by zero")
    return nominator / denominator


def nGramMatchingIntern(h, r, n):
    count = 0
    lHyp = fetchNGrams(h, n)
    lRef = fetchNGrams(r, n)
    for hyp in lHyp:
        for ref in lRef:
            if hyp == ref:
                count += 1
                lRef.remove(ref)
                break
    return count


def fetchNGrams(seq, n):
    gramlist = []
    for i in range(0, len(seq) - n + 1):
        gramlist.append(seq[i:i + n])  # i is max. len(h) - n => we're slicing max. to len(h) - 1
    return gramlist


def nGramMatching(h, r, n):
    return nGramMatchingIntern(h.split(), r.split(), n)


def brevityPenalty(data: list[tuple[str, str]]):
    c = r = 0
    for (ref, hyp) in data:
        c += len(hyp.split())
        r += len(ref.split())

    if c > r:
        return 1
    elif c:
        return np.exp(1 - (r / c))
    else: # c == 0
        return 0


def bleu4(data: list[tuple[str, str]]):
    bp = brevityPenalty(data)
    s = 0
    for i in range(1, 5):
        s += (1 / 4) * np.log(nGramPrecision(data, i))
    return bp * np.exp(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref")
    parser.add_argument("--hyp")
    parser.add_argument("--metric")
    args = parser.parse_args()

    ref_in = open(args.ref, 'r', encoding="utf-8")
    #ref_lines2 = [line.split() for line in ref_in]
    ref_lines = ref_in.readlines() #!!! somehow readlines changes ref_in

    ref_in.close()

    hyp_in = open(args.hyp, 'r', encoding="utf-8")
    #hyp_lines2 = [line.split() for line in hyp_in]
    #print(sum([len(x) for x in hyp_lines2]))
    hyp_lines = hyp_in.readlines()
    hyp_in.close()

    data = list(zip(ref_lines, hyp_lines))
    #data2 = list(zip(ref_lines2, hyp_lines2))

    match args.metric:
        case "WER":
            print(WER(data))
            #print(WER2(data2))
            #metric = WordErrorRate()
            #metric.update(ref_lines, hyp_lines)
            #print(metric.compute())
            #print(wer(hyp_lines, ref_lines))
        case "PER":
            print(PER(data))
        case "BLEU":
            print(bleu4(data))
            #metric = BLEUScore(n_gram=4)
            #hyp_lines_lists = [[x] for x in hyp_lines]
            #metric.update(ref_lines, hyp_lines_lists)
            print(bleu(hyp_lines, ref_lines))

def main2():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref")
    parser.add_argument("--hyp")
    parser.add_argument("--metric")
    args = parser.parse_args()

    ref_in = open(args.ref, 'r', encoding="utf-8")
    ref_lines = [line.split() for line in ref_in]
    ref_in.close()

    hyp_in = open(args.hyp, 'r', encoding="utf-8")
    hyp_lines = [line.split() for line in hyp_in]
    hyp_in.close()

    data = list(zip(ref_lines, hyp_lines))

    match args.metric:
        case "WER":
            print(WER2(data))
            torchwer = wer([' '.join(sublist) for sublist in hyp_lines], [' '.join(sublist) for sublist in ref_lines])
            print("Torch WER: ", str(torchwer))
        case "PER":
            print(PER(data))
        case "BLEU":
            print(bleu4(data))
            #metric = BLEUScore(n_gram=4)
            #metric.update(ref_lines, hyp_lines)
            #print(metric.compute())


if __name__ == "__main__":
    main2()
