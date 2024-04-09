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


def nGramMatchingIntern(h, r, n):
    count = 0
    hwrapper = [h]
    break_flag = False
    while len(r) >= n:
        # sequence to match
        l1 = r[0:n]
        # iterate over hypothesis
        for j in range(0, len(hwrapper)):
            if len(hwrapper[j]) >= n:
                for i in range(0, len(hwrapper[j]) - n + 1):
                    # possible matching sequence
                    l2 = hwrapper[j][i:i + n]  # i is max. len(h) - n => we're slicing max. to len(h) - 1
                    if l1 == l2:
                        count += 1
                        # first n words get "popped" in r => won't be used again (pop n-1 words first, the n'th will be
                        # popped after break)
                        r = r[n - 1:]
                        # the match in h gets popped, splitting h => prevents formation of new sequences
                        hwrapper.insert(j+1, hwrapper[j][i + n:])
                        hwrapper[j] = hwrapper[j][:i]
                        # breaking the outer for loop
                        break_flag = True
                        break
            if break_flag:
                break_flag = False
                break
        r = r[1:]
    return count


def nGramMatching(h, r, n):
    return nGramMatchingIntern(h.split(), r.split(), n)


# levenhsteinWrapper("hallo hallo ja","hallo ja hallo")
print(nGramMatching("and the one only", "I am the one and only", 2))
