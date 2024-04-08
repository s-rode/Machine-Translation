import numpy as np
def levenshtein(a, b):
    lenA = len(a)+1
    lenB = len(b)+1
    if lenA == 0 or lenB == 0:
        print("either a or b are empty")
    #init as zeros
    matr = np.zeros([lenA, lenB], dtype=int)

    #fill first column
    for i in range(lenA):
        matr[i][0] = i

    #fill first row
    for j in range(lenB):
        matr[0][j] = j

    for i in range(1, lenA):
        for j in range(1, lenB):
            if a[i-1] == b[j-1]:
                matr[i][j] = matr[i-1][j-1]
            else:
                matr[i][j] = min(matr[i-1][j-1],matr[i][j-1],matr[i-1][j])+1

    i = lenA-1
    j = lenB-1
    sol = []
    while (i > 0) or (j > 0):
        if(matr[i-1][j] == matr[i][j]-1):
            sol.append(f"insert {a[i-1]}")
            i = i-1
        elif(matr[i][j-1] == matr[i][j]-1):
            sol.append(f"delete {b[j-1]}")
            j = j-1
        elif (matr[i-1][j-1] == matr[i][j]):
            i = i - 1
            j = j - 1
            sol.append("match")
        else:
            sol.append(f"substitute {a[i - 1]} with {b[j - 1]}")
            i = i - 1
            j = j - 1
    print("Weg",sol)
    print(matr[lenA-1][lenB-1])

def levenhsteinWrapper(a,b):
    levenshtein(a.split(),b.split())

#levenhsteinWrapper("hallo hallo ja","hallo ja hallo")
levenshtein("ananas","banane")
