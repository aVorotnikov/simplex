import numpy as np


def parse_file(filename):
    n = 0
    mode = 'max'
    A = []
    b = np.array([])
    indexes_not_bounded = []
    with open(filename, 'r') as file:
        for line in file:
            if '=' not in line:
                indexes_not_bounded = {i for i in range(n)} - {int(x) - 1 for x in line.split()}
                continue
            sep = '='
            if '<' in line:
                sep = '<='
            elif '>' in line:
                sep = '>='
            line_split = line.split(sep)
            if len(line_split) != 2:
                return None
            lhs, rhs = line_split
            lhs = lhs.strip().lower()
            rhs = rhs.strip().lower()
            if lhs == 'n':
                if not rhs.isdigit() or int(rhs) <= 0 or sep != '=':
                    return None
                n = int(rhs)
            elif lhs == 'mode':
                mode = rhs
                if rhs not in ('min', 'max') or sep != '=':
                    return None
            elif lhs == 'c':
                if sep != '=':
                    return None
                c = np.array([float(coef) for coef in rhs.split()])
                if mode == 'min':
                    c *= -1
            else:
                a = np.array([float(coef) for coef in lhs.split()])
                bi = float(rhs)
                if sep == '>=':
                    a *= -1
                    bi *= -1
                elif sep == '=':
                    A.append(-a)
                    b = np.append(b, -bi)
                A.append(a)
                b = np.append(b, bi)

    A = np.array(A)
    for i in indexes_not_bounded:
        c = np.append(c, -c[i])
        A = np.append(A, -A.take(i, axis=1).reshape(len(b), 1), axis=1)

    return A, b, c


def init_canonical(A, b, c):
    n = len(c) + len(b)
    N = {i for i in range(len(c))}
    B = {i for i in range(len(c), n)}
    v = 0
    Ac = np.zeros((n, n))
    bc = np.zeros((n, 1))
    cc = np.zeros((n, 1))

    for i, ii in enumerate(range(len(c), n)):
        bc[ii] = b[i]
        cc[i] = c[i]
        for j, jj in enumerate(range(len(c))):
            Ac[ii][jj] = A[i][j]

    return N, B, Ac, bc, cc, v
