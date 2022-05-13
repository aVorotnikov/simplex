import numpy as np


def pivot(N, B, A, b, c, v, l, e):
    A_new = np.zeros(A.shape)
    b_new = np.zeros(b.shape)
    c_new = np.zeros(c.shape)
    b_new[e] = b[l] / A[l][e]
    for j in N - {e}:
        A_new[e][j] = A[l][j] / A[l][e]
    A_new[e][l] = 1 / A[l][e]

    for i in B - {l}:
        b_new[i] = b[i] - A[i][e] * b_new[e]
        for j in N - {e}:
            A_new[i][j] = A[i][j] - A[i][e] * A_new[e][j]
        A_new[i][l] = -A[i][e] * A_new[e][l]

    v_new = v + c[e] * b_new[e]

    for j in N - {e}:
        c_new[j] = c[j] - c[e] * A_new[e][j]
    c_new[l] = -c[e] * A_new[e][l]

    N.remove(e)
    N.add(l)
    B.remove(l)
    B.add(e)

    return N, B, A_new, b_new, c_new, v_new


def build_canonical(A, b, c):
    m, n = A.shape
    N = {i for i in range(n)}
    B = {i for i in range(n, n + m)}
    v = 0
    Ac = np.zeros((n + m, n + m))
    bc = np.zeros(n + m)
    cc = np.zeros(n + m)

    for i in range(n):
        cc[i] = c[i]

    for i in range(m):
        bc[n + i] = b[i]
        for j in range(n):
            Ac[n + i][j] = A[i][j]

    return N, B, Ac, bc, cc, v


def init_simplex(A_in, b_in, c_in):
    k = np.argmin(b_in)
    if b_in[k] >= 0:
        return build_canonical(A_in, b_in, c_in)
    else:
        caux = np.zeros((len(c_in) + 1, 1))
        caux[0] = -1

        Aaux = np.zeros((len(b_in), len(c_in) + 1))
        for i in range(len(b_in)):
            Aaux[i][0] = -1
            for j in range(len(c_in)):
                Aaux[i][j + 1] = A_in[i][j]

        N, B, A, b, c, v = build_canonical(Aaux, b_in, caux)
        l = len(c_in) + k + 1
        N, B, A, b, c, v = pivot(N, B, A, b, c, v, l, 0)
        while any([c[j] > 0 for j in N]):
            e = 0
            while c[e] <= 0 or e not in N:
                e += 1
            delta = np.full((len(b),), np.inf)
            for i in B:
                delta[i] = b[i] / A[i][e] if A[i][e] > 1e-14 else np.inf
            l = np.argmin(delta)
            if delta[l] == np.inf:
                print('Задача неограничена')
                return None
            else:
                assert (l in B)
                N, B, A, b, c, v = pivot(N, B, A, b, c, v, l, e)

        x = np.zeros(b.shape)
        for i in B:
            x[i] = b[i]
        if abs(x[0]) < 1e-14:
            c = np.zeros(A.shape[1])
            for i in range(1, len(c_in) + 1):
                if i in N:
                    c[i] += c_in[i - 1]
                else:  # in B
                    for j in range(A.shape[1]):
                        c[j] -= c_in[i - 1] * A[i][j]
                    v += c_in[i - 1] * b[i]
            N = {i - 1 for i in N if i != 0}
            B = {i - 1 for i in B if i != 0}
            return N, B, A[1:, 1:], b[1:], c[1:], v
        else:
            print('Задача неразрешима')
            return None


def simplex_table(A_input, b_input, c_input):
    canonical = init_simplex(A_input, b_input, c_input)
    #canonical = build_canonical(A_input, b_input, c_input)
    if not canonical:
        return None
    N, B, A, b, c, v = canonical
    while any([c[j] > 0 for j in N]):
        e = 0
        while c[e] <= 0 or e not in N:
            e += 1
        delta = np.full((len(b),), np.inf)
        for i in B:
            delta[i] = (b[i] / A[i][e]) if A[i][e] > 1e-14 else np.inf
        l = np.argmin(delta)
        if delta[l] == np.inf:
            print('Задача неограниченна')
            return None
        else:
            assert(l in B)
            N, B, A, b, c, v = pivot(N, B, A, b, c, v, l, e)

    x = np.zeros(c.shape)
    for i in B:
        x[i] = b[i]
    return x[:len(x) // 2], v
