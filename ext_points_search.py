import numpy as np
from itertools import combinations
import simplex as sx

def get_basis_matrices(A):
    N = A.shape[0]
    M = A.shape[1]

    basis_matrices = []
    basis_combinations_indexes = []
    all_indexes = [i for i in range(M)]

    for i in combinations(all_indexes, N):
        basis_matrix = A[:, i]
        if np.linalg.det(basis_matrix) != 0:
            basis_matrices.append(basis_matrix)
            basis_combinations_indexes.append(i)

    return basis_matrices, basis_combinations_indexes


def get_vectors(A, b):
    N = len(A[0])
    M = len(A)
    vectors = []

    if M >= N:
        return vectors
    else:
        basis_matrices, basis_combinations_indeces = get_basis_matrices(np.array(A))

    for i in range(len(basis_matrices)):
        solve = np.linalg.solve(basis_matrices[i], b)
        if (len(solve[solve < 0]) != 0) or (len(solve[solve > 1e+15]) != 0):
            continue

        vec = [0 for i in range(N)]
        for j in range(len(basis_combinations_indeces[i])):
            vec[basis_combinations_indeces[i][j]] = solve[j]
        vectors.append(vec)
    return 

def solve_brute_force(A, b, c, v):
    vectors = get_vectors(A, b)
    if len(vectors) == 0:
        return []

    solution = vectors[0]
    target_max = np.dot(solution, c)

    for vec in vectors:
        if np.dot(vec, c) > target_max:
            target_max = np.dot(vec, c)
            solution = vec

    return solution

def extreme_points_search(N, B, A, b, c, v):
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
print(extreme_points_search(N, B, A, b, c, v))
