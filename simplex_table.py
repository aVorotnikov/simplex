import numpy as np
import simplex as sx

def pivot(N, B, A, b, c, v, l, e):
    A_ = np.zeros(A.shape)
    b_ = np.zeros(b.shape)
    c_ = np.zeros(c.shape)
    b_[e] = b[l] / A[l][e]
    for j in N - {e}:
            A_[e][j] = A[l][j] / A[l][e]
    A[e][l] = 1 / A[l][e]

    for i in B - {l}:
        b_[i] = b[i] - A[i][e] * b_[e]
        for j in N - {e}:
            A_[i][j] = A[i][j] - A[i][e] * A_[e][j]
        A_[i][l] = -A[i][e] * A_[e][l]

    v_ = v + c[e] * b_[e]
    for j in N - {e}:
        c_[j] = c[j] - c[e] * A_[e][j]
    c_[l] = -c[e] * A_[e][l]

    N.remove(e)
    N.add(l)
    B.remove(l)
    B.add(e)

    return N, B, A_, b_, c_, v_


def simplex_table(N, B, A, b, c, v):
    while any([c[j] > 0 for j in N]):
        e = 0
        while c[e] <= 0 or e not in N:
            e += 1
        delta = np.full(b.shape, np.Inf)
        for i in B:
            delta[i] = b[i] / A[i][e] if A[i][e] > 0 else np.Inf
        l = np.argmin(delta)
        if delta[l] is np.Inf:
            return None
        else:
            N, B, A, b, c, v = pivot(N, B, A, b, c, v, l, e)

    x = np.zeros(b.shape)
    for i in B:
        x[i] = b[i]
    return x, v


A, b, c = sx.parse_file('test.txt')
N, B, A, b, c, v = sx.init_canonical(A, b, c)
print(simplex_table(N, B, A, b, c, v))
