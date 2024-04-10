import numpy as np


def levenshtein(a, b):
    lenA = len(a) + 1
    lenB = len(b) + 1
    if lenA == 0 or lenB == 0:
        print("either a or b are empty")
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

    i = lenA - 1
    j = lenB - 1
    sol = []
    while (i > 0) or (j > 0):
        if (matr[i - 1][j] == matr[i][j] - 1):
            sol.append(f"insert {a[i - 1]}")
            i = i - 1
        elif (matr[i][j - 1] == matr[i][j] - 1):
            sol.append(f"delete {b[j - 1]}")
            j = j - 1
        elif (matr[i - 1][j - 1] == matr[i][j]):
            i = i - 1
            j = j - 1
            sol.append("match")
        else:
            sol.append(f"substitute {a[i - 1]} with {b[j - 1]}")
            i = i - 1
            j = j - 1
    solStr = "; ".join(sol)
    print("Lösungsweg: ", solStr)
    print(matr[lenA - 1][lenB - 1])


def levenhsteinWrapper(a, b):
    levenshtein(a.split(), b.split())


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
    list = []
    for i in range(0, len(seq) - n + 1):
        list.append(seq[i:i + n])  # i is max. len(h) - n => we're slicing max. to len(h) - 1
    return list


def nGramMatching(h, r, n):
    return nGramMatchingIntern(h.split(), r.split(), n)


def brevityPenalty(data: list[tuple[str, str]]):
    c = r = 0
    for (ref, hyp) in data:
        c += hyp.split()
        r += ref.split()

    if c > r:
        return 1
    else:
        return np.exp(1 - (r / c))


def bleu4(data: list[tuple[str, str]]):
    bp = brevityPenalty(data)
    s = 0
    for i in range(1, 5):
        s += (1 / 4) * np.log(nGramPrecision(data, i))
    return bp * np.exp(s)

# levenhsteinWrapper("hallo hallo ja","hallo ja hallo")


# print(nGramPrecision([("the cat is on the mat", "the cat")], 1))
# print(fetchNGrams(["hallo","test","hallo","h"],2))
